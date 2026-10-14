"""Core library: task construction, hidden-state extraction, held-out-word probes and readout.

Representation/readout gate (the core probing library used by every runner):
  A. Knowledge-prediction GAP: per layer, can a linear probe recover the answer
     (KNOWLEDGE) while the model's own logit-lens readout / final output cannot
     (PREDICTION)? Large gap = "answer present but not routed to output."
  B. Dual-position probing: KNOWLEDGE at the word-token position (presence — char
     info lives here, ~L22 from Stage A) vs at the answer position (readout). The
     word->answer drop is the ROUTING failure our intervention targets.
  C. Suppression localization: per-layer logit-lens prob of the correct answer.
  D. Causal injection sanity (hooks): move the present word-position direction to the
     answer position at a late layer; does the correct-answer prob rise?

Tasks: routing-type (first_letter, last_letter, contains_letter) where a read-out fix
is apt; compute-type contrast (extraction at arbitrary position, count with variance).
Completion-style prompts -> the answer is the next token.

    uv run python scripts/run_stage1.py --device cuda
    uv run python scripts/run_stage1.py --smoke
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ExperimentConfig, apply_env_overrides
from src.model import CausalLM, load_tokenizer
from src.data import CHAR_PROBE_WORDS_BY_LEN, _ordinal
from src.intervention import residual_hooks, add_direction, final_norm, lm_head

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage1")

ROUTING = ["first_letter", "last_letter", "contains_letter"]
COMPUTE = ["extraction", "counting"]
ALPHA = "abcdefghijklmnopqrstuvwxyz"

# Copy-vs-synthesize battery: dissociates "answer is a present token (a letter the model
# emits fluently)" from "answer must be synthesized (a count -> digit)", holding difficulty
# roughly matched. most_frequent_letter = COPY but compute-heavy (harder than counting);
# word_length = SYNTH but easy. If alignment tracks copy-vs-synth (not difficulty),
# most_frequent aligns w/ scale while word_length/count_distinct stay stuck.
CVS_TASKS = ["most_frequent_letter", "letter_before", "word_length", "count_distinct"]
# LENGTH-MATCHED count tasks (v4, E6). Same questions as counting / count_distinct, but examples
# are drawn from a (word length x answer) rectangle with equal counts per cell, so word length is
# statistically independent of the answer by construction (the v3 count stratification made
# length predict the count; see the nuisance-floor section of the paper).
LM_TASKS = ["counting_lm", "count_distinct_lm"]
# answer type per task: letters (copy) vs digits (synth) vs yes/no
LETTER_TASKS = {"first_letter", "last_letter", "extraction", "most_frequent_letter", "letter_before"}
DIGIT_TASKS = {"counting", "word_length", "count_distinct", "counting_lm", "count_distinct_lm"}
TASK_SETS = {"orig": ROUTING + COMPUTE, "cvs": CVS_TASKS, "lm": LM_TASKS}

# v3 iterated Python sets in the sampler, so (word, query-letter) draws depended on
# PYTHONHASHSEED. v4 iterates sorted sets. ORTHO_LEGACY_SET_ORDER=1 restores the v3 stream (only
# useful for reproducing the published v3 grid under a matching hash seed).
LEGACY_SET_ORDER = os.environ.get("ORTHO_LEGACY_SET_ORDER") == "1"


def _letters(w):
    """Distinct letters of `w` in a process-independent order (see LEGACY_SET_ORDER)."""
    return list(set(w)) if LEGACY_SET_ORDER else sorted(set(w))


def base_task(task):
    """`counting_lm` -> `counting`: the length-matched tasks share the base task's question."""
    return task[:-3] if task.endswith("_lm") else task


def first_content_token(tokenizer, s):
    """First NON-space token of ' '+s. Tokenizer-agnostic: letters/yes/no are usually a
    single token (' a'); digits often split as [<space>, digit], so if the first token
    decodes to whitespace we take the next one. Works across Qwen/Llama/Gemma tokenizers."""
    ids = tokenizer(" " + s, add_special_tokens=False).input_ids
    if not ids:
        return tokenizer.unk_token_id
    if len(ids) > 1 and tokenizer.decode([ids[0]]).strip() == "":
        return ids[1]
    return ids[0]


def model_tag(name):
    return name.split("/")[-1].replace(".", "_").replace("-", "_")


def load_word_pool(words_file=None, min_len=3):
    """The stimulus word pool. With --words-file (e.g. src/stimulus_words.txt, ~6k frequency-ranked
    words) the pool is large enough that the answer probe cannot memorize a word->answer lookup;
    without it, falls back to the small built-in CHAR_PROBE_WORDS_BY_LEN pool (legacy)."""
    if words_file:
        ws = [w.strip().lower() for w in open(words_file) if w.strip()]
        ws = [w for w in ws if w.isalpha() and len(w) >= min_len]
        if not ws:
            raise ValueError(f"no usable words in {words_file}")
        return sorted(set(ws))
    return sorted({w for ws in CHAR_PROBE_WORDS_BY_LEN.values() for w in ws if len(w) >= min_len})


def train_logreg_gpu(H, labels, k, device, l2=1e-3, max_iter=500):
    """GPU multinomial logistic regression -> raw-space weight matrix W (d x k),
    aligned to label index (columns for absent classes = 0); None if <2 usable classes.

    Standardizes features for conditioning, fits full-batch LBFGS on the GPU (converges
    cheaply -> no CPU-lbfgs slowness on the many-class letter probes), then folds the
    scaling back so W applies to RAW hidden states (matching how the firm uses coef.T,
    with the per-class intercept dropped). Moves the probe off the 6 vCPU onto the A6000."""
    present, counts = np.unique(labels, return_counts=True)
    present = present[counts >= 3]
    if len(present) < 2:
        return None
    keep = np.isin(labels, present)
    remap = {int(c): i for i, c in enumerate(present)}
    X = torch.tensor(H[keep], dtype=torch.float32, device=device)
    y = torch.tensor([remap[int(v)] for v in labels[keep]], dtype=torch.long, device=device)
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, keepdim=True) + 1e-6
    Xn = (X - mu) / sd
    P = len(present)
    W = torch.zeros(X.shape[1], P, device=device, requires_grad=True)
    b = torch.zeros(P, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, history_size=10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xn @ W + b, y) + l2 * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    Wraw = (W.detach() / sd.t())  # raw-space (d, P); per-class constant dropped (as coef.T)
    out = torch.zeros(X.shape[1], k, device=device)
    out[:, torch.tensor([int(c) for c in present], device=device)] = Wraw
    return out.detach().cpu().numpy()


def last_real_index(attention_mask):
    """Index of the LAST real (non-pad) token per row, correct for BOTH padding sides.

    ``attention_mask.sum(1) - 1`` is only correct for RIGHT padding; with left padding it
    points into the middle of the sequence (the Gemma-2 padding bug, Appendix C). Padding
    is always a contiguous run at one end, so the last real token is the last position whose
    mask is 1 -> found by locating the first 1 from the right. Returns a (B,) long tensor.
    """
    L = attention_mask.shape[1]
    rev = torch.flip(attention_mask, dims=[1])
    return (L - 1 - rev.long().argmax(dim=1)).long()


def bootstrap_ci_by_word(vals, words, n_boot=1000, seed=0, alpha=0.05):
    """Word-level bootstrap CI for a mean (or a per-example difference of means).

    `vals` is a per-example float array (NaN = example not scored); `words` the query word of
    each example. We resample UNIQUE WORDS with replacement (the unit of generalization for a
    held-out-word probe) and recompute the mean over all examples of the drawn words. Returns
    {mean, lo, hi, n_words, n_boot} or None. Review item §7.15."""
    vals = np.asarray(vals, dtype=float)
    words = np.asarray(words)
    ok = ~np.isnan(vals)
    if ok.sum() < 2:
        return None
    vals, words = vals[ok], words[ok]
    uniq = np.unique(words)
    idx_by_word = {w: np.where(words == w)[0] for w in uniq}
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        draw = uniq[rng.randint(0, len(uniq), len(uniq))]
        idx = np.concatenate([idx_by_word[w] for w in draw])
        boots.append(float(vals[idx].mean()))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": round(float(vals.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "n_words": int(len(uniq)), "n_boot": int(n_boot)}


def even_layers(n_hidden, n_probe):
    """`n_probe` evenly spaced layer indices over [0, n_hidden-1] (inclusive of both ends).

    Fixes the depth-dependent layer-selection bias of an ABSOLUTE --probe-stride, which gives
    16-layer models 5 draws at the max and 40-layer models 11. Always includes
    layer 0 and the final layer, so every model gets the same number of draws."""
    n_probe = max(2, min(int(n_probe), n_hidden))
    return sorted(set(int(round(x)) for x in np.linspace(0, n_hidden - 1, n_probe)))


def probe_acc_gpu(H, y, device, min_per=5, folds=3, max_iter=200, l2=1e-3, warm_start=True):
    """GPU knowledge-probe ACCURACY (drop-in for the CPU probe_acc): drops classes with
    < min_per samples, fits a standardized multinomial logreg (LBFGS) on a 70/30 split,
    averages test accuracy over `folds` splits. Moves the per-layer probe off the 6 vCPU
    onto the GPU — the dominant gate cost for many-class letter probes. Returns
    {acc, n_classes, chance} or None."""
    classes, counts = np.unique(y, return_counts=True)
    keep = classes[counts >= min_per]
    if len(keep) < 2:
        return None
    mask = np.isin(y, keep)
    remap = {int(c): i for i, c in enumerate(keep)}
    yi = np.array([remap[int(v)] for v in y[mask]])
    Xall = torch.tensor(H[mask], dtype=torch.float32, device=device)
    yall = torch.tensor(yi, dtype=torch.long, device=device)
    n, d, k = len(yi), H.shape[1], len(keep)
    if n < 24:
        return None
    accs, majs = [], []
    W_prev = b_prev = None  # warm-start across folds: folds overlap 70%, so the optima are
    for f in range(folds):   # near-identical -> LBFGS early-stops in a few iters after fold 0
        perm = np.random.RandomState(f).permutation(n)
        cut = max(2, int(0.7 * n))
        tr_idx, te_idx = perm[:cut], perm[cut:]
        tr = torch.tensor(tr_idx, device=device)
        te = torch.tensor(te_idx, device=device)
        if len(te) == 0:
            continue
        Xtr, ytr, Xte, yte = Xall[tr], yall[tr], Xall[te], yall[te]
        mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
        Xtrn, Xten = (Xtr - mu) / sd, (Xte - mu) / sd
        W = (W_prev.clone() if W_prev is not None else torch.zeros(d, k, device=device)).requires_grad_(True)
        b = (b_prev.clone() if b_prev is not None else torch.zeros(k, device=device)).requires_grad_(True)
        opt = torch.optim.LBFGS([W, b], max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(Xtrn @ W + b, ytr) + l2 * (W * W).sum()
            loss.backward()
            return loss
        opt.step(closure)
        if warm_start:
            W_prev, b_prev = W.detach(), b.detach()
        with torch.no_grad():
            accs.append(((Xten @ W + b).argmax(1) == yte).float().mean().item())
        modal = np.bincount(yi[tr_idx]).argmax()               # majority-class baseline (numpy: MPS-safe)
        majs.append(float(np.mean(yi[te_idx] == modal)))
    if not accs:
        return None
    return {"acc": float(np.mean(accs)), "n_classes": k, "chance": 1.0 / k,
            "majority": float(np.mean(majs)) if majs else 1.0 / k}


def _fit_eval_std(Xtr, ytr, Xte, yte, k, device, max_iter=200, l2=1e-3, W0=None, b0=None):
    """Standardized multinomial logreg (LBFGS). Returns (test_acc, W, b, correct); pass W0/b0 to
    warm-start (grouped folds overlap ~(K-1)/K, so LBFGS converges in a few iters after fold 0).
    `correct` is the per-test-example 0/1 correctness vector (numpy), used for word-level
    bootstrap CIs and held-out layer selection."""
    d = Xtr.shape[1]
    mu = Xtr.mean(0, keepdim=True); sd = Xtr.std(0, keepdim=True) + 1e-6
    Xtrn, Xten = (Xtr - mu) / sd, (Xte - mu) / sd
    W = (W0.clone() if W0 is not None else torch.zeros(d, k, device=device)).requires_grad_(True)
    b = (b0.clone() if b0 is not None else torch.zeros(k, device=device)).requires_grad_(True)
    opt = torch.optim.LBFGS([W, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xtrn @ W + b, ytr) + l2 * (W * W).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        ok = ((Xten @ W + b).argmax(1) == yte).float()
        acc = ok.mean().item()
    return acc, W.detach(), b.detach(), ok.cpu().numpy()


def probe_acc_grouped(H, y, groups, device, min_per=5, folds=5, max_iter=200, l2=1e-3,
                      warm_start=True, return_preds=False):
    """Leakage-safe knowledge probe: K folds over UNIQUE groups (query words) so no word is in
    both train and test. Guards against a high-capacity probe memorizing a word->answer lookup
    over the small word pool. Same estimator as probe_acc_gpu. Also returns the majority-class
    baseline (predict the train modal class) so grouped accuracy can be read against a frequency
    prior, not just uniform chance. Returns {acc, n_classes, chance, majority, folds, n_words}."""
    classes, counts = np.unique(y, return_counts=True)
    keep = classes[counts >= min_per]
    if len(keep) < 2:
        return None
    mask = np.isin(y, keep)
    remap = {int(c): i for i, c in enumerate(keep)}
    yi = np.array([remap[int(v)] for v in y[mask]])
    g = np.asarray(groups)[mask]
    Xall = torch.tensor(H[mask], dtype=torch.float32, device=device)
    yall = torch.tensor(yi, dtype=torch.long, device=device)
    k = len(keep)
    uniq = np.array(sorted(set(g.tolist())))
    if len(uniq) < folds:
        folds = max(2, len(uniq))
    order = np.random.RandomState(0).permutation(len(uniq))
    fold_of = {w: (i % folds) for i, w in enumerate(uniq[order])}
    gfold = np.array([fold_of[w] for w in g])
    accs, majs = [], []
    # per-example held-out correctness (each example is in exactly ONE test fold); NaN = not scored
    n_all = len(yi)
    ex_correct = np.full(n_all, np.nan)
    ex_maj = np.full(n_all, np.nan)
    fold_accs, fold_majs, fold_ids = [], [], []
    W_prev = b_prev = None  # warm-start across folds (they overlap (K-1)/K of the data)
    for f in range(folds):
        te_m = gfold == f
        tr_m = ~te_m
        if te_m.sum() == 0 or tr_m.sum() < 24:
            continue
        if len(np.unique(yi[tr_m])) < 2 or len(np.unique(yi[te_m])) < 1:
            continue
        tr = torch.tensor(np.where(tr_m)[0], device=device)
        te = torch.tensor(np.where(te_m)[0], device=device)
        acc, W_new, b_new, ok = _fit_eval_std(Xall[tr], yall[tr], Xall[te], yall[te], k, device,
                                              max_iter, l2, W_prev, b_prev)
        if warm_start:
            W_prev, b_prev = W_new, b_new
        accs.append(acc)
        modal = np.bincount(yi[tr_m]).argmax()                 # majority baseline (numpy: MPS-safe)
        maj = float(np.mean(yi[te_m] == modal))
        majs.append(maj)
        te_idx = np.where(te_m)[0]
        ex_correct[te_idx] = ok
        ex_maj[te_idx] = (yi[te_idx] == modal).astype(float)
        fold_accs.append(acc); fold_majs.append(maj); fold_ids.append(f)
    if not accs:
        return None
    out = {"acc": float(np.mean(accs)), "n_classes": k, "chance": 1.0 / k,
           "majority": float(np.mean(majs)) if majs else 1.0 / k,
           "folds": len(accs), "n_words": int(len(uniq))}
    if return_preds:
        out.update({"kept_mask": mask, "ex_correct": ex_correct, "ex_majority": ex_maj,
                    "ex_fold": gfold, "fold_accs": fold_accs, "fold_majs": fold_majs,
                    "fold_ids": fold_ids})
    return out


def _stem(task, w, pos=None, c=None):
    task = base_task(task)
    if task == "first_letter":
        return f'The first letter of the word "{w}" is'
    if task == "last_letter":
        return f'The last letter of the word "{w}" is'
    if task == "extraction":
        return f'The {_ordinal(pos)} letter of the word "{w}" is'
    if task == "contains_letter":
        return f'Does the word "{w}" contain the letter "{c}"? Answer yes or no:'
    if task == "most_frequent_letter":
        return f'In the word "{w}", the letter that appears most often is'
    if task == "letter_before":
        return f'In the word "{w}", the letter immediately before "{c}" is'
    if task == "word_length":
        return f'The word "{w}" has this many letters:'
    if task == "count_distinct":
        return f'The word "{w}" has this many different letters:'
    return f'In the word "{w}", the letter "{c}" appears this many times:'  # counting


def _sample_task(task, w, rng):
    """Return (stem, answer) for `task` on word `w`, or None if `w` is unsuitable
    (e.g. a tie for most_frequent_letter, or no valid before-letter). Used for both
    query examples and few-shot demos so their formats are identical."""
    if task.endswith("_lm"):           # demos for a length-matched task = ordinary base items
        return _sample_task(base_task(task), w, rng)
    if task == "first_letter":
        return _stem(task, w), w[0]
    if task == "last_letter":
        return _stem(task, w), w[-1]
    if task == "extraction":
        p = rng.randint(1, len(w))
        return _stem(task, w, pos=p), w[p - 1]
    if task == "contains_letter":
        if rng.random() < 0.5:
            c, a = rng.choice(_letters(w)), "yes"
        else:
            ab = [x for x in ALPHA if x not in w]
            c = rng.choice(ab) if ab else rng.choice(_letters(w))
            a = "yes" if c in w else "no"
        return _stem(task, w, c=c), a
    if task == "counting":
        c = rng.choice(_letters(w))
        return _stem(task, w, c=c), str(w.count(c))
    if task == "most_frequent_letter":  # COPY + compute (argmax over letter counts)
        cnt = {ch: w.count(ch) for ch in _letters(w)}
        mx = max(cnt.values())
        top = [ch for ch, k in cnt.items() if k == mx]
        if len(top) != 1:
            return None                 # skip ties -> unambiguous label
        return _stem(task, w), top[0]
    if task == "letter_before":         # COPY (answer is a present letter)
        cand = [i for i, ch in enumerate(w) if i > 0 and w.count(ch) == 1]
        if not cand:
            return None
        i = rng.choice(cand)
        return _stem(task, w, c=w[i]), w[i - 1]
    if task == "word_length":           # SYNTH + easy
        if len(w) > 9:
            return None                 # keep single-digit answers
        return _stem(task, w), str(len(w))
    if task == "count_distinct":        # SYNTH + compute
        d = len(set(w))
        if d > 9:
            return None
        return _stem(task, w), str(d)
    raise ValueError(f"unknown task {task}")


def _demo(task, words, rng, avoid):
    """A worked example (stem, answer) of `task` on a word != avoid."""
    for _ in range(20):
        w = rng.choice(words)
        if w == avoid:
            continue
        r = _sample_task(task, w, rng)
        if r:
            return r
    return _sample_task(task, avoid, rng) or (_stem(task, avoid), avoid[0])


def _wrap(stem_s, ans, demos, tokenizer):
    """Build the few-shot prompt. Space-split answers (digits: ' 2'->[<space>,'2'])
    get a trailing space on the stem so the CONTENT token is the immediate next
    prediction; letters/yes-no are single tokens and need no trailing space."""
    if tokenizer is not None:
        ids = tokenizer(" " + ans, add_special_tokens=False).input_ids
        if len(ids) > 1 and tokenizer.decode([ids[0]]).strip() == "":
            stem_s = stem_s + " "
    lines = [f"{s} {a}." for s, a in demos]
    return "\n".join(lines + [stem_s]) if lines else stem_s


LM_MIN_ITEMS = 20   # min distinct items per (length, answer) cell for the length-matched design
LM_REPETITION = 2   # counting_lm: only words with len(w) - #distinct letters == 2
LM_MIN_PER_LETTER = 2   # counting_lm: a query letter needs >= this many items at EVERY count


def _lm_items(task, words):
    """{(word length, answer): [(word, query letter or None), ...]} for a length-matched task.

    counting_lm removes three word-only shortcuts found in the v3/v4 nuisance floors:
      length       -> equal examples per (length, count) cell (lm_design)
      repetition   -> only words with exactly LM_REPETITION repeated letters, so "no repeats =>
                      count 1" is unavailable (length - #distinct was a 0.59 predictor)
      query letter -> only letters that occur at every count at that length; build_examples then
                      draws the letter uniformly per cell ('e' alone predicted 3 at 0.46)."""
    cells = defaultdict(list)
    for w in words:
        if task == "counting_lm":
            if len(w) - len(set(w)) != LM_REPETITION:
                continue
            for ch in _letters(w):
                c = w.count(ch)
                if c in (1, 2, 3):
                    cells[(len(w), c)].append((w, ch))
        elif task == "count_distinct_lm":
            d = len(set(w))
            if d <= 9:
                cells[(len(w), d)].append((w, None))
        else:
            raise ValueError(f"not a length-matched task: {task}")
    if task == "counting_lm":                  # keep only letters available at every count
        for L in {L for L, _ in cells}:
            per = [Counter(ch for _, ch in cells.get((L, c), [])) for c in (1, 2, 3)]
            ok = {ch for ch in per[0] if all(p[ch] >= LM_MIN_PER_LETTER for p in per)}
            for c in (1, 2, 3):
                if (L, c) in cells:
                    cells[(L, c)] = [it for it in cells[(L, c)] if it[1] in ok]
        cells = defaultdict(list, {k: v for k, v in cells.items() if v})
    return cells


def lm_design(task, words, min_items=LM_MIN_ITEMS):
    """The (lengths, answers) rectangle for a length-matched task: contiguous word lengths x
    contiguous answers (>= 3 answers, >= 2 lengths) in which EVERY cell has >= `min_items`
    items. Prefers more answer classes, then more lengths. Deterministic given `words`."""
    cells = _lm_items(task, words)
    Ls = sorted({L for L, _ in cells})
    As = sorted({a for _, a in cells})
    best, best_score = None, None
    for i in range(len(Ls)):
        for j in range(i + 1, len(Ls)):
            LL = Ls[i:j + 1]
            for p in range(len(As)):
                for q in range(p + 2, len(As)):
                    AA = As[p:q + 1]
                    if all(len(cells.get((L, a), [])) >= min_items for L in LL for a in AA):
                        score = (len(AA), len(LL), -i)
                        if best_score is None or score > best_score:
                            best, best_score = (LL, AA), score
    if best is None:
        raise ValueError(f"no length-matched rectangle with >= {min_items} items/cell for {task}")
    return best


def build_examples(words, n, seed, shots=3, tokenizer=None, tasks=None):
    """Few-shot char-task prompts. `shots`=0 reproduces the old zero-shot format;
    `shots`>=1 prepends worked examples so the answer is a reliable next token across
    model sizes (the format-transfer fix). `tasks` selects which tasks to generate
    (default = the original 5)."""
    tasks = tasks or (ROUTING + COMPUTE)
    rng = random.Random(seed)
    ex = []

    def emit(task, w, stem_s, ans):
        demos = [_demo(task, words, rng, w) for _ in range(shots)]
        ex.append(dict(task=task, word=w, answer=ans,
                       prompt=_wrap(stem_s, ans, demos, tokenizer)))

    for task in tasks:
        if task in LM_TASKS:
            # LENGTH-MATCHED: equal examples per (word length, answer) cell over a rectangle, so
            # length carries no information about the answer (E6).
            cells = _lm_items(task, words)
            lens, answers = lm_design(task, words)
            per = max(1, n // (len(lens) * len(answers)))
            for L in lens:
                for a in answers:
                    items = cells[(L, a)]
                    if task == "counting_lm":   # query letter uniform within the cell
                        by_letter = defaultdict(list)
                        for it in items:
                            by_letter[it[1]].append(it)
                        letters = sorted(by_letter)
                        picks = [rng.choice(by_letter[rng.choice(letters)]) for _ in range(per)]
                    else:
                        picks = (rng.sample(items, per) if len(items) >= per
                                 else [rng.choice(items) for _ in range(per)])
                    for w, ch in picks:
                        emit(task, w, _stem(task, w, c=ch), str(a))
            continue
        if task == "counting":
            # COUNT VARIANCE: stratify queried (word,letter) by true count
            pool = defaultdict(list)
            for w in words:
                for ch in _letters(w):
                    pool[w.count(ch)].append((w, ch))
            cs = [c for c in (1, 2, 3) if len(pool[c]) >= 5]
            per = max(1, n // max(1, len(cs)))
            for c in cs:
                for _ in range(per):
                    w, ch = rng.choice(pool[c])
                    emit("counting", w, _stem("counting", w, c=ch), str(c))
            continue
        made = 0
        for _ in range(n * 6):          # retry budget (some words unsuitable per task)
            if made >= n:
                break
            w = rng.choice(words)
            r = _sample_task(task, w, rng)
            if r is None:
                continue
            stem_s, ans = r
            emit(task, w, stem_s, ans)
            made += 1
    return ex


def answer_space_ids(tokenizer, task):
    chars = list(ALPHA) if task in LETTER_TASKS else \
        (["yes", "no"] if task == "contains_letter" else list("0123456789"))
    return [first_content_token(tokenizer, c) for c in chars]


@torch.no_grad()
def extract(model, tokenizer, examples, device, batch_size=64, keep_layers=None, want_word=True,
            word_mean=False):
    """Per example: hidden at the ANSWER position (last real token) and (if want_word) the WORD
    position (last token overlapping the word span) at the requested layers; final logits; target
    answer token id.

    Efficiency (Tier A): the final logits are computed from the final-layer answer-position hidden
    (head(norm(h))), so we pass logits_to_keep=1 to skip the full-vocabulary LM-head matmul over
    every position -- a large saving for big-vocab models (Gemma 256k, Qwen 150k). want_word=False
    skips the offset-mapping + word-position gather entirely (used with --skip-word-probe).
    word_mean=True (requires want_word) also returns `word_mean_h`: the MEAN over every token that
    overlaps the query word's span, so letters carried by a multi-token word's earlier tokens are
    not lost to a last-token read (v4, E1)."""
    n_hidden = model.base_model.config.num_hidden_layers + 1
    layers = list(range(n_hidden)) if keep_layers is None else sorted(keep_layers)
    ans_h = {l: [] for l in layers}
    word_h = {l: [] for l in layers}
    wmean_h = {l: [] for l in layers}
    word_ntok = []
    final_logits, target_ids, tasks = [], [], []
    bm = model.base_model
    norm, head = final_norm(model), lm_head(model)
    hd = next(head.parameters()).dtype
    for s in range(0, len(examples), batch_size):
        batch = examples[s:s + batch_size]
        prompts = [e["prompt"] for e in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=192, return_offsets_mapping=want_word)
        offsets = enc.pop("offset_mapping").tolist() if want_word else None
        enc = enc.to(device)
        try:  # logits_to_keep=1: don't waste the LM head on every position (we recompute below)
            out = bm(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                     output_hidden_states=True, logits_to_keep=1)
        except TypeError:  # a custom/remote forward that doesn't accept the kwarg
            out = bm(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                     output_hidden_states=True)
        # last REAL token, correct for either padding side (was attention_mask.sum(1)-1, which
        # silently mis-indexed left-padding tokenizers like Gemma-2's)
        last_idx = last_real_index(enc.attention_mask)
        rows = torch.arange(len(batch), device=device)
        for l in layers:  # gather only the needed answer position on-GPU, copy once
            ans_h[l].append(out.hidden_states[l][rows, last_idx].float().cpu().numpy())
        if want_word:
            wi_list = []
            span_w = torch.zeros(enc.input_ids.shape, dtype=torch.float32, device=device)
            for b, e in enumerate(batch):
                li = int(last_idx[b]); w, pr = e["word"], e["prompt"]
                st = pr.rfind('"' + w + '"')  # query word = LAST occurrence (after few-shot demos)
                wi = li
                if st >= 0:
                    a, z = st + 1, st + 1 + len(w)
                    cand = [p for p, (o0, o1) in enumerate(offsets[b]) if o1 > a and o0 < z and o1 > 0]
                    if cand:
                        wi = cand[-1]
                        span_w[b, cand] = 1.0 / len(cand)
                if span_w[b].sum() == 0:
                    span_w[b, wi] = 1.0
                word_ntok.append(int((span_w[b] > 0).sum().item()))
                wi_list.append(wi)
            wi_t = torch.tensor(wi_list, device=device)
            for l in layers:
                word_h[l].append(out.hidden_states[l][rows, wi_t].float().cpu().numpy())
                if word_mean:
                    wm = torch.einsum("bt,btd->bd", span_w, out.hidden_states[l].float())
                    wmean_h[l].append(wm.cpu().numpy())
        # final logits = head(norm(final-layer answer hidden)) == the model's logits at the answer
        # position, without the full-sequence LM-head matmul
        fh = out.hidden_states[-1][rows, last_idx].to(hd)
        final_logits.append(head(norm(fh)).float().cpu().numpy())
        for e in batch:
            target_ids.append(first_content_token(tokenizer, e["answer"]))
            tasks.append(e["task"])
    return {"ans_h": {l: np.concatenate(ans_h[l]) for l in layers},
            "word_h": {l: np.concatenate(word_h[l]) for l in layers} if want_word else {},
            "word_mean_h": ({l: np.concatenate(wmean_h[l]) for l in layers}
                            if (want_word and word_mean) else {}),
            "word_ntok": np.array(word_ntok) if want_word else None,
            "final_logits": np.concatenate(final_logits),
            "target_ids": np.array(target_ids), "tasks": np.array(tasks)}


def probe_acc(H, y, min_per=5):
    """CV accuracy of a linear probe predicting the answer class. Drops classes with
    < min_per samples (so one rare letter doesn't kill the whole probe). Returns
    (acc, n_classes, n_kept) or None if too sparse."""
    classes, counts = np.unique(y, return_counts=True)
    keep = set(classes[counts >= min_per])
    if len(keep) < 2:
        return None
    mask = np.isin(y, list(keep))
    H, y = H[mask], y[mask]
    if len(y) < 24:
        return None
    k = int(min(5, np.unique(y, return_counts=True)[1].min()))
    if k < 2:
        return None
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    acc = float(cross_val_score(clf, H, y, cv=cv, scoring="accuracy", n_jobs=-1).mean())
    return {"acc": acc, "n_classes": len(keep), "n": int(mask.sum()), "chance": 1.0 / len(keep)}


def heldout_layer_select(fold_tab, fold_ids):
    """Statistically-correct max-over-layers: for each held-out fold f, pick the layer that is best
    on the OTHER folds and report fold f's accuracy at that layer; average over folds. Removes the
    selection optimism of `max over layers of the mean`.

    `fold_tab[L] = {fold_id: (acc, maj)}`. Returns {acc, majority, layers} or None."""
    layers = [L for L in fold_tab if fold_tab[L]]
    if not layers or not fold_ids:
        return None
    accs, majs, chosen = [], [], {}
    for f in fold_ids:
        best_L, best_v = None, -1.0
        for L in layers:
            others = [fold_tab[L][g][0] for g in fold_ids if g != f and g in fold_tab[L]]
            if not others:
                continue
            v = float(np.mean(others))
            if v > best_v:
                best_v, best_L = v, L
        if best_L is None or f not in fold_tab[best_L]:
            continue
        a, mj = fold_tab[best_L][f]
        accs.append(a); majs.append(mj); chosen[int(f)] = int(best_L)
    if not accs:
        return None
    return {"acc": float(np.mean(accs)), "majority": float(np.mean(majs)), "layers": chosen}


def probe_arm(H_by_layer, y, groups, device, layers, folds=3, warm_start=True, heldout=False,
              n_boot=0, boot_seed=0):
    """Held-out-word probe of label `y` from `H_by_layer[L]` at every layer in `layers`.

    With the defaults this reproduces run_noquery/run_crosstask.probe_condition exactly (peak
    over layers of the fold-mean, majority from the peak layer). v4 adds, without changing
    those fields:
      heldout=True   `acc_heldout_sel` / `above_majority_heldout_sel`: for each fold, the layer
                     chosen on the OTHER folds (removes max-over-layers selection optimism).
      warm_start=False  cold-start every fold.
      n_boot>0       word-level bootstrap 95% CI on above-majority at the peak layer.
    Returns a dict, or None if no layer could be probed."""
    by_layer, fold_tab, best = {}, {}, None
    need_preds = heldout or n_boot > 0
    for L in layers:
        if L not in H_by_layer:
            continue
        r = probe_acc_grouped(H_by_layer[L], y, groups, device, folds=folds,
                              warm_start=warm_start, return_preds=need_preds)
        if not r:
            continue
        by_layer[int(L)] = round(r["acc"], 4)
        if heldout:
            fold_tab[int(L)] = {int(f): (float(a), float(m)) for f, a, m in
                                zip(r["fold_ids"], r["fold_accs"], r["fold_majs"])}
        if best is None or r["acc"] >= best[1]["acc"]:
            best = (int(L), r)
    if best is None:
        return None
    L, r = best
    maj = float(r.get("majority", r["chance"]))
    out = {"acc_peak": by_layer[L], "peak_layer": L, "majority": round(maj, 4),
           "above_majority": round(by_layer[L] - maj, 4), "chance": round(r["chance"], 3),
           "n_classes": r["n_classes"], "n_words": r["n_words"], "folds": r["folds"],
           "by_layer": by_layer}
    if heldout:
        ids = sorted({f for d in fold_tab.values() for f in d})
        hs = heldout_layer_select(fold_tab, ids)
        if hs:
            out["acc_heldout_sel"] = round(hs["acc"], 4)
            out["majority_heldout_sel"] = round(hs["majority"], 4)
            out["above_majority_heldout_sel"] = round(hs["acc"] - hs["majority"], 4)
            out["heldout_layers"] = hs["layers"]
    if n_boot > 0 and "ex_correct" in r:
        g = np.asarray(groups)[r["kept_mask"]]
        ci = bootstrap_ci_by_word(r["ex_correct"] - r["ex_majority"], g, n_boot, boot_seed)
        if ci:
            out["above_majority_ci95"] = ci
    return out


def gate_metrics(model, tokenizer, data, task_list, device, ex_words,
                 probe_stride=4, probe_folds=3, group_probe=False, group_folds=5,
                 skip_word_probe=True, smoke=False, layers=None, layer_select="max",
                 n_boot=0, warm_start=True, boot_seed=0):
    """Assemble the representation/readout gate metrics from an already-extracted `data`. Shared by
    main() and the unified run_grid.py so one forward pass serves gate + alignment. Returns a dict
    with the per-task `gap` and the by-layer probe/lens dicts (same schema main writes).

    Controls (all OFF by default -> byte-identical behaviour to the published grid):
      layers        explicit list of layer indices to probe (see even_layers()); overrides stride.
      layer_select  "max" (peak over layers, as published) or "heldout" (additionally report a
                    held-out-fold layer choice; the peak fields are still written).
      n_boot        >0 -> word-level bootstrap CIs on prediction_acc and above-majority.
      warm_start    False -> cold-start every CV fold (retires the warm-start concern, §6.3).
    """
    n_hidden = len(data["ans_h"])
    tasks, targets = data["tasks"], data["target_ids"]
    norm, head = final_norm(model), lm_head(model)
    hd = next(head.parameters()).dtype
    space_ids = {t: answer_space_ids(tokenizer, t) for t in task_list}
    ex_words = np.asarray(ex_words)

    prediction_acc, pred_correct = {}, {}
    for t in task_list:
        m = tasks == t
        if not m.any():
            continue
        sp = np.array(space_ids[t])
        pred = sp[data["final_logits"][m][:, sp].argmax(axis=1)]
        prediction_acc[t] = float(np.mean(pred == targets[m]))
        pred_correct[t] = (pred == targets[m]).astype(float)

    lens_acc, lens_prob, know_ans, know_word, know_meta = (defaultdict(dict) for _ in range(5))
    if layers:
        probe_layers = sorted({int(l) for l in layers if 0 <= int(l) < n_hidden})
    else:
        stride = 4 if smoke else max(1, probe_stride)
        probe_layers = sorted(set(range(0, n_hidden, stride)) | {n_hidden - 1})
    folds = 2 if smoke else probe_folds
    want_preds = bool(group_probe and (n_boot > 0 or layer_select == "heldout"))
    fold_tab = defaultdict(lambda: defaultdict(dict))   # task -> layer -> {fold: (acc, maj)}
    peak_preds = {}                                     # task -> per-example correctness at peak L
    for l in probe_layers:
        Ha = torch.tensor(data["ans_h"][l]).to(device)
        with torch.no_grad():
            logits = head(norm(Ha.to(hd))).float().cpu().numpy()
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        for t in task_list:
            m = tasks == t
            if m.sum() == 0:
                continue
            sp = np.array(space_ids[t])
            pred = sp[logits[m][:, sp].argmax(axis=1)]
            lens_acc[t][l] = float(np.mean(pred == targets[m]))
            lens_prob[t][l] = float(np.mean(probs[m][np.arange(m.sum()), targets[m]]))
            if group_probe:
                ka = probe_acc_grouped(data["ans_h"][l][m], targets[m], ex_words[m], device,
                                       folds=group_folds, warm_start=warm_start,
                                       return_preds=want_preds)
            else:
                ka = probe_acc_gpu(data["ans_h"][l][m], targets[m], device, folds=folds,
                                   warm_start=warm_start)
            if ka:
                know_ans[t][l] = ka["acc"]
                know_meta[t]["answer"] = {"n_classes": ka["n_classes"], "chance": round(ka["chance"], 3),
                                          "majority": round(ka.get("majority", ka["chance"]), 3)}
                if want_preds and "fold_accs" in ka:
                    for fi, fa, fm in zip(ka["fold_ids"], ka["fold_accs"], ka["fold_majs"]):
                        fold_tab[t][l][int(fi)] = (float(fa), float(fm))
                    if know_ans[t][l] >= max(know_ans[t].values()):
                        # per-example held-out correctness at the current best layer, mapped back
                        # onto the task's example indexing (NaN where the class was dropped)
                        n_t = int(m.sum())
                        c = np.full(n_t, np.nan); mj = np.full(n_t, np.nan)
                        kept = np.where(ka["kept_mask"])[0]
                        c[kept] = ka["ex_correct"]; mj[kept] = ka["ex_majority"]
                        peak_preds[t] = {"layer": l, "correct": c, "majority": mj}
            if not skip_word_probe and data.get("word_h"):
                kw = probe_acc_gpu(data["word_h"][l][m], targets[m], device, folds=folds,
                                   warm_start=warm_start)
                if kw:
                    know_word[t][l] = kw["acc"]

    def peak(d):
        return (max(d.values()), max(d, key=d.get)) if d else (0.0, None)

    gap = {}
    for t in task_list:
        ka, kal = peak(know_ans[t])
        kw, kwl = peak(know_word[t])
        la, lal = peak(lens_acc[t])
        chance = 1.0 / len(set(space_ids[t]))
        pacc = prediction_acc.get(t, 0.0)
        maj = (know_meta[t].get("answer") or {}).get("majority", chance)
        gap[t] = {
            "chance": round(chance, 3),
            "majority": round(maj, 3),
            "grouped": bool(group_probe),
            "prediction_acc": round(pacc, 4),
            "lens_peak": round(la, 4), "lens_peak_layer": lal,
            "knowledge_answer_peak": round(ka, 4), "knowledge_answer_layer": kal,
            "knowledge_above_majority": round(ka - maj, 4),
            "knowledge_word_peak": round(kw, 4) if not skip_word_probe else None,
            "knowledge_word_layer": kwl,
            "probe_classes": know_meta[t].get("answer"),
            "gap_answer_vs_pred": round(ka - pacc, 4),
            "gap_word_vs_answer": round(kw - ka, 4) if not skip_word_probe else None,
        }
        # --- held-out layer selection (unbiased alternative to the peak) ---
        if layer_select == "heldout" and fold_tab.get(t):
            ids = sorted({f for L in fold_tab[t] for f in fold_tab[t][L]})
            hs = heldout_layer_select(fold_tab[t], ids)
            if hs:
                gap[t]["knowledge_answer_heldout_layer_sel"] = round(hs["acc"], 4)
                gap[t]["above_majority_heldout_layer_sel"] = round(hs["acc"] - hs["majority"], 4)
                gap[t]["heldout_layer_choice"] = hs["layers"]
        # --- word-level bootstrap CIs ---
        if n_boot > 0:
            ci = {}
            mt = tasks == t
            w_t = ex_words[mt]
            if t in pred_correct:
                c = bootstrap_ci_by_word(pred_correct[t], w_t, n_boot, boot_seed)
                if c:
                    ci["prediction_acc"] = c
            pp = peak_preds.get(t)
            if pp is not None:
                c = bootstrap_ci_by_word(pp["correct"], w_t, n_boot, boot_seed)
                if c:
                    ci["knowledge_answer_peak"] = c
                d = bootstrap_ci_by_word(pp["correct"] - pp["majority"], w_t, n_boot, boot_seed)
                if d:
                    ci["knowledge_above_majority"] = {**d, "layer": int(pp["layer"])}
            if ci:
                gap[t]["ci95"] = ci
    return {"gap": gap, "prediction_acc": prediction_acc,
            "probe_layers": probe_layers,
            "knowledge_answer_by_layer": {t: know_ans[t] for t in know_ans},
            "knowledge_word_by_layer": {t: know_word[t] for t in know_word},
            "logit_lens_acc_by_layer": {t: lens_acc[t] for t in lens_acc},
            "logit_lens_correct_prob_by_layer": {t: lens_prob[t] for t in lens_prob}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=400, help="examples per task")
    ap.add_argument("--out", default="outputs/stage1_gate/results.json")
    ap.add_argument("--model", default=None, help="override model name (for the scale ladder)")
    ap.add_argument("--shots", type=int, default=3, help="few-shot demos (0=zero-shot; the format ablation)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--probe-stride", type=int, default=2,
                    help="probe every Nth layer (peak is smooth across layers; 2 halves cost)")
    ap.add_argument("--probe-folds", type=int, default=3, help="CV splits for the GPU probe")
    ap.add_argument("--probe-layers", type=int, default=0,
                    help="CONTROL: probe a FIXED NUMBER of evenly-spaced layers (e.g. 12) instead "
                         "of an absolute stride, so max-over-layers is comparable across depths")
    ap.add_argument("--layer-select", default="max", choices=["max", "heldout"],
                    help="CONTROL: 'heldout' additionally reports a fold-held-out layer choice")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="CONTROL: word-level bootstrap resamples for 95%% CIs (0=off)")
    ap.add_argument("--no-warm-start", action="store_true",
                    help="CONTROL: cold-start every CV fold (retires the warm-start concern)")
    ap.add_argument("--skip-word-probe", action="store_true",
                    help="skip the word-position probe (halves probe cost); rep-vs-readout only needs know@ans vs pred")
    ap.add_argument("--no-remote-code", action="store_true",
                    help="load with trust_remote_code=False (force native arch class; e.g. Phi-4-mini whose remote modeling_phi3.py needs a newer transformers)")
    ap.add_argument("--set", dest="taskset", default="orig", choices=list(TASK_SETS),
                    help="task set: 'orig' (5 tasks) or 'cvs' (copy-vs-synthesize battery)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="forward batch size (lower for big models at high shots, e.g. 13B s5 OOM)")
    ap.add_argument("--group-probe", action="store_true",
                    help="leakage-safe REPRESENTATION: probe knowledge_answer with GroupKFold BY "
                         "QUERY WORD (no word in train+test) + majority-class baseline. Writes to "
                         "outputs/stage1_gate_grouped[_set]/. Use for the leakage-safe main grid.")
    ap.add_argument("--group-folds", type=int, default=5, help="held-out-word folds when --group-probe")
    ap.add_argument("--words-file", default=None,
                    help="large stimulus word list (e.g. src/stimulus_words.txt, ~6k words) -> the "
                         "leakage-safe primary pool; outputs go to bigstim_* dirs. Omit for the "
                         "legacy ~100-word built-in pool.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="if the output file already exists, skip (cheap restarts / fill missing cells)")
    args = ap.parse_args()

    task_list = TASK_SETS[args.taskset]
    config = ExperimentConfig()
    mt = f"_{model_tag(args.model)}" if args.model else ""
    if args.model:  # scale ladder: results tagged per model so rungs don't collide
        config.model.model_name = args.model
    # output dir: [bigstim_]stage1_gate[_grouped][_cvs]/
    base = "stage1_gate_grouped" if args.group_probe else "stage1_gate"
    if args.taskset != "orig":
        base = f"{base}_{args.taskset}"
    if args.words_file:
        base = "bigstim_" + base
    args.out = args.out.replace("stage1_gate/", base + "/")
    args.out = args.out.replace(".json", f"{mt}_s{args.shots}.json")  # tag model + shot count
    if args.skip_existing and Path(args.out).exists():
        logger.info("skip (exists): %s", args.out); return
    apply_env_overrides(config)
    if args.no_remote_code:
        config.model.trust_remote_code = False
    logger.info("Loading model %s on %s", config.model.model_name, args.device)
    model = CausalLM(config.model).to(args.device)
    model.eval()
    tokenizer = load_tokenizer(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    words = load_word_pool(args.words_file)
    logger.info("word pool: %d words (%s)", len(words), args.words_file or "built-in ~100-word pool")
    if args.smoke:
        args.n = 60
    examples = build_examples(words, args.n, args.seed, shots=args.shots,
                              tokenizer=tokenizer, tasks=task_list)
    ex_words = np.array([e["word"] for e in examples])  # query word per example (--group-probe)
    by_task = defaultdict(int)
    for e in examples:
        by_task[e["task"]] += 1
    logger.info("examples: %d %s", len(examples), dict(by_task))

    data = extract(model, tokenizer, examples, args.device, batch_size=args.batch_size,
                   want_word=not args.skip_word_probe)
    n_hidden = len(data["ans_h"])
    tasks, targets = data["tasks"], data["target_ids"]
    all_tasks = task_list

    sel_layers = even_layers(n_hidden, args.probe_layers) if args.probe_layers else None
    metrics = gate_metrics(model, tokenizer, data, task_list, args.device, ex_words,
                           probe_stride=args.probe_stride, probe_folds=args.probe_folds,
                           group_probe=args.group_probe, group_folds=args.group_folds,
                           skip_word_probe=args.skip_word_probe, smoke=args.smoke,
                           layers=sel_layers, layer_select=args.layer_select,
                           n_boot=args.bootstrap, warm_start=not args.no_warm_start)
    gap = metrics["gap"]
    prediction_acc = metrics["prediction_acc"]
    know_ans = metrics["knowledge_answer_by_layer"]
    know_word = metrics["knowledge_word_by_layer"]
    lens_acc = metrics["logit_lens_acc_by_layer"]
    lens_prob = metrics["logit_lens_correct_prob_by_layer"]

    # D. causal injection sanity on first_letter: move the present word-position
    # first-letter direction to the answer position at late layers. (only for task sets
    # that include first_letter + run the word probe; skipped for the cvs battery)
    inject = {}
    read_layer = 22
    if "first_letter" in all_tasks and not args.skip_word_probe and know_word.get("first_letter"):
        read_layer = gap["first_letter"]["knowledge_word_layer"] or 22
        m = tasks == "first_letter"
        Wread = data["word_h"][read_layer][m]
        yfl = targets[m]
        gmean = Wread.mean(axis=0)
        dirs = {int(tid): Wread[yfl == tid].mean(axis=0) - gmean
                for tid in np.unique(yfl) if (yfl == tid).sum() >= 5}
        test = [e for e in examples if e["task"] == "first_letter"][:60]
        for write in sorted({min(read_layer + d, n_hidden - 2) for d in (2, 4, 6)}):
            bp, ip = [], []
            for e in test:
                tid = tokenizer(" " + e["answer"], add_special_tokens=False).input_ids[0]
                if tid not in dirs:
                    continue
                enc = tokenizer(e["prompt"], return_tensors="pt").to(args.device)
                li = int(enc.attention_mask.sum()) - 1
                with torch.no_grad():
                    bp.append(float(torch.softmax(model.base_model(**enc).logits[0, li], -1)[tid]))
                    vec = torch.tensor(dirs[tid])
                    with residual_hooks(model, {write - 1: add_direction(vec, 4.0, positions=[li])}):
                        ip.append(float(torch.softmax(model.base_model(**enc).logits[0, li], -1)[tid]))
            if bp:
                inject[f"write_layer_{write}"] = {"n": len(bp),
                    "base": round(float(np.mean(bp)), 4), "injected": round(float(np.mean(ip)), 4),
                    "lift": round(float(np.mean(ip) - np.mean(bp)), 4)}

    results = {"gap": gap, "prediction_acc": prediction_acc,
               "knowledge_answer_by_layer": {t: know_ans[t] for t in know_ans},
               "knowledge_word_by_layer": {t: know_word[t] for t in know_word},
               "logit_lens_acc_by_layer": {t: lens_acc[t] for t in lens_acc},
               "logit_lens_correct_prob_by_layer": {t: lens_prob[t] for t in lens_prob},
               "injection_sanity": inject,
               "meta": {"model": config.model.model_name, "n_per_task": args.n,
                        "n_layers": n_hidden, "read_layer": read_layer,
                        "smoke": args.smoke, "seed": args.seed, "set": args.taskset,
                        "grouped": bool(args.group_probe), "words_file": args.words_file,
                        "n_words": len(words)}}
    out = Path(args.out if not args.smoke else args.out.replace(".json", "_smoke.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 74)
    print("STAGE I — causal-feasibility gate   (chance | model | lens | know@ans | know@word)")
    print("=" * 74)
    for t in all_tasks:
        g = gap[t]
        kind = "ROUTING" if t in ROUTING else "compute"
        print(f"\n{t:16s} [{kind}]  chance={g['chance']}  probe_classes={g['probe_classes']}")
        kwp = g['knowledge_word_peak']
        rgap = g['gap_word_vs_answer']
        kw_str = f"{kwp:.3f}@{g['knowledge_word_layer']}" if kwp is not None else "skipped"
        rgap_str = f"{rgap:+.3f}" if rgap is not None else "n/a (word probe skipped)"
        print(f"  pred_acc={g['prediction_acc']:.3f}  lens_peak={g['lens_peak']:.3f}@{g['lens_peak_layer']}  "
              f"know@ans={g['knowledge_answer_peak']:.3f}@{g['knowledge_answer_layer']}  "
              f"know@word={kw_str}")
        print(f"  READOUT gap (know@ans - pred) = {g['gap_answer_vs_pred']:+.3f}   "
              f"ROUTING gap (know@word - know@ans) = {rgap_str}")
    print("\ninjection sanity (first_letter, correct-letter prob; word-dir -> answer pos):")
    for k, v in inject.items():
        print(f"  {k}: base={v['base']:.3f} -> {v['injected']:.3f}  lift={v['lift']:+.3f} (n={v['n']})")
    print("\nGATE: large READOUT gap (know@ans >> pred) => KAPPA-style alignment applies (best).")
    print("      else large ROUTING gap (know@word >> know@ans) => must MOVE info (harder).")
    print("      injection lift > 0 => routable. Both gaps ~0 => info not usable => reframe.")
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
