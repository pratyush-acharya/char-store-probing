"""Nuisance baselines for the character knowledge-prediction gap (LOCAL, no GPU, no model).

WHY
---
The paper baselines every probe against a MAJORITY-CLASS baseline. That controls for label
skew only. It does *not* control for nuisance features that (a) the model demonstrably encodes
almost perfectly and (b) are correlated with the label by construction of the stimulus set.
Two such features:

  1. WORD LENGTH.  `build_examples` stratifies `counting` by the true count (1/2/3) to balance
     classes; longer words are much more likely to contain a letter 3 times. So length alone
     predicts the count. The models encode length at ~0.96-1.00 (the paper's own `word_length`
     task), so a "probe" that reads length off the residual stream and maps it to a count is
     doing no character computation at all.
  2. LETTER-SET MEMBERSHIP (position-free).  Which letters occur in the word, with no ordering
     and no multiplicity. For `first_letter`/`last_letter`/`extraction` the answer is *some*
     member of that set, so a position-free predictor that learns the train answer-prior over
     letters already gets a lot.

This script reconstructs each task EXACTLY as `scripts/run_stage1.build_examples` does
(n=800, seed=42, tokenizer=None, words from src/stimulus_words.txt) and fits held-out-WORD
predictors that see ONLY a nuisance feature, using the SAME fold construction as
`run_stage1.probe_acc_grouped` (classes with <5 support dropped; folds over
sorted(set(words)) permuted with RandomState(0), fold = i % folds; folds=3 as in run_grid.py).
It then reads the model numbers out of outputs/bigstim_stage1_gate_grouped{,_cvs}/ and reports
above-majority next to above-NUISANCE.

ORACLE WARNING (read the flags in the output): some nuisance features are not nuisances at all
for some tasks -- they are sufficient statistics for the label. `len(word)` IS the answer for
`word_length`; the binary letter set determines `count_distinct` exactly (it is the popcount);
the query letter's membership determines `contains_letter`. Those cells are marked ORACLE and
must NOT be read as "the model result is confounded" -- they say the task is trivially solvable
from the feature, which is a statement about the task, not about the probe. The load-bearing
cells are the ones marked NUISANCE.

DETERMINISM CAVEAT (found while writing this): `run_stage1._sample_task` calls
`rng.choice(list(set(w)))` for `counting`/`contains_letter` and iterates `set(w)` for
`most_frequent_letter`. Python randomizes `str.__hash__` per process, so SET ITERATION ORDER --
and therefore the exact (word, query-letter) draws for those tasks -- depends on PYTHONHASHSEED.
The grid's cloud run is thus not bit-reproducible for those four tasks. Distributional
quantities are stable (our reconstructed majority matches every recorded JSON to <=0.0005, i.e.
to the JSONs' rounding), but individual nuisance accuracies move by ~+-0.02 across hash seeds.
This script therefore re-execs itself with PYTHONHASHSEED=0 so its own numbers are reproducible;
`--hash-seeds N` quantifies the spread.

Run:  uv run python scripts/analyze_nuisance_baselines.py
      uv run python scripts/analyze_nuisance_baselines.py --hash-seeds 5
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_stage1 as s1  # noqa: E402  (task construction ONLY -- no model is loaded)

from sklearn.linear_model import LogisticRegression  # noqa: E402

ALPHA = s1.ALPHA
WORDS_FILE = str(ROOT / "src" / "stimulus_words.txt")
N_PER_TASK = 800
SEED = 42
FOLDS = 3          # run_grid.py --group-folds default
MIN_PER = 5        # run_stage1.probe_acc_grouped min_per
SHOTS = 3          # headline shot setting

GEMMA = {"gemma_2_2b", "gemma_2_9b"}

# tasks whose stem contains a query character (available to the model, so a fair nuisance
# predictor may see it too)
QUERY_CHAR_TASKS = {"contains_letter", "counting", "letter_before"}

# feature -> set of tasks for which that feature is a SUFFICIENT statistic for the label
ORACLE = {
    "length": {"word_length"},
    "letterset": {"count_distinct"},
    "letterset+q": {"count_distinct", "contains_letter"},
    "length+letterset": {"word_length", "count_distinct"},
}

TASKS_ORIG = s1.TASK_SETS["orig"]
TASKS_CVS = s1.TASK_SETS["cvs"]
ALL_TASKS = TASKS_ORIG + TASKS_CVS
TASK_SET_OF = {t: ("orig" if t in TASKS_ORIG else "cvs") for t in ALL_TASKS}


# --------------------------------------------------------------------------- task rebuild
def build(shots):
    """Rebuild both task batteries exactly as the grid did. Note: `shots` changes the RNG
    stream (each emit() draws `shots` demos), so the example set is shot-specific."""
    words = s1.load_word_pool(WORDS_FILE)
    ex = []
    for ts in ("orig", "cvs"):
        ex += s1.build_examples(words, N_PER_TASK, SEED, shots=shots,
                                tokenizer=None, tasks=s1.TASK_SETS[ts])
    by = defaultdict(list)
    for e in ex:
        by[e["task"]].append(e)
    return by


def query_char(task, e):
    """Recover the query letter from the stem (the tasks that have one)."""
    m = re.findall(r'"([a-z])"', e["prompt"].split("\n")[-1])
    return m[-1] if m else None


# --------------------------------------------------------------------------- fold machinery
def prepare(rows):
    """Apply the probe's class filter and build the held-out-word folds.
    Mirrors run_stage1.probe_acc_grouped exactly."""
    y = np.array([e["answer"] for e in rows])
    g = np.array([e["word"] for e in rows])
    cnt = Counter(y.tolist())
    keep = {k for k, v in cnt.items() if v >= MIN_PER}
    m = np.array([v in keep for v in y])
    y, g, rows = y[m], g[m], [r for r, ok in zip(rows, m) if ok]
    uniq = np.array(sorted(set(g.tolist())))
    folds = FOLDS if len(uniq) >= FOLDS else max(2, len(uniq))
    order = np.random.RandomState(0).permutation(len(uniq))
    fold_of = {w: (i % folds) for i, w in enumerate(uniq[order])}
    gfold = np.array([fold_of[w] for w in g])
    return rows, y, g, gfold, folds, sorted(keep)


def cv_score(rows, y, gfold, folds, fit_predict):
    """Average test accuracy of `fit_predict(train_rows, train_y, test_rows)` over folds,
    plus the per-fold majority (train modal class evaluated on test) for cross-checking."""
    accs, majs = [], []
    for f in range(folds):
        te = gfold == f
        tr = ~te
        if te.sum() == 0 or tr.sum() < 24:
            continue
        if len(set(y[tr])) < 2:
            continue
        rtr = [r for r, ok in zip(rows, tr) if ok]
        rte = [r for r, ok in zip(rows, te) if ok]
        pred = fit_predict(rtr, y[tr], rte)
        accs.append(float(np.mean(np.asarray(pred) == y[te])))
        modal = Counter(y[tr].tolist()).most_common(1)[0][0]
        majs.append(float(np.mean(y[te] == modal)))
    return float(np.mean(accs)), float(np.mean(majs))


# --------------------------------------------------------------------------- the predictors
def make_majority():
    def f(rtr, ytr, rte):
        modal = Counter(ytr.tolist()).most_common(1)[0][0]
        return [modal] * len(rte)
    return f


def make_length_only():
    """Per-length train-modal answer; global train modal for unseen lengths."""
    def f(rtr, ytr, rte):
        by = defaultdict(Counter)
        for r, a in zip(rtr, ytr):
            by[len(r["word"])][a] += 1
        glob_modal = Counter(ytr.tolist()).most_common(1)[0][0]
        table = {L: c.most_common(1)[0][0] for L, c in by.items()}
        return [table.get(len(r["word"]), glob_modal) for r in rte]
    return f


def _bag(rows, with_query, task):
    X = np.zeros((len(rows), 26 + (26 if with_query else 0)), dtype=np.float64)
    for i, r in enumerate(rows):
        for ch in set(r["word"]):
            if ch in ALPHA:
                X[i, ALPHA.index(ch)] = 1.0
        if with_query:
            c = query_char(task, r)
            if c:
                X[i, 26 + ALPHA.index(c)] = 1.0
    return X


def make_letterset(task, with_query=False, with_length=False):
    """Multinomial logistic regression on a position-free 26-dim binary bag-of-letters
    (optionally + query-letter one-hot, optionally + word length)."""
    def f(rtr, ytr, rte):
        Xtr, Xte = _bag(rtr, with_query, task), _bag(rte, with_query, task)
        if with_length:
            Xtr = np.hstack([Xtr, np.array([[len(r["word"])] for r in rtr], float)])
            Xte = np.hstack([Xte, np.array([[len(r["word"])] for r in rte], float)])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit((Xtr - mu) / sd, ytr)
        return clf.predict((Xte - mu) / sd)
    return f


def make_length_only_regr():
    """length + letterset via the modal-per-length table is weak; this is the combined
    logistic model (length as a continuous feature alongside the bag)."""
    return make_letterset("", with_query=False, with_length=True)


# --------------------------------------------------------------------------- model results
def load_gate(shots):
    """{(model, task): gap-dict} from the two 6000-word grouped gate dirs."""
    out = {}
    for sub in ("bigstim_stage1_gate_grouped", "bigstim_stage1_gate_grouped_cvs"):
        for fp in glob.glob(str(ROOT / "outputs" / sub / f"results_*_s{shots}.json")):
            m = re.match(r".*results_(.+)_s\d\.json", fp)
            model = m.group(1)
            for t, g in json.load(open(fp))["gap"].items():
                out[(model, t)] = g
    return out


# --------------------------------------------------------------------------- main
def hash_seed_sweep(nseeds):
    """Re-run this script's TABLE A1 under several PYTHONHASHSEEDs to quantify how much the
    set-iteration-order nondeterminism in run_stage1._sample_task moves the baselines."""
    accum = defaultdict(lambda: defaultdict(list))
    for hs in range(nseeds):
        env = dict(os.environ, PYTHONHASHSEED=str(hs), _NUIS_CHILD="1")
        out = subprocess.run([sys.executable, __file__, "--emit-json"], env=env,
                             capture_output=True, text=True, cwd=str(ROOT))
        d = json.loads(out.stdout.strip().splitlines()[-1])
        for t, r in d.items():
            for k in ("majority", "length", "letterset", "letterset+q", "length+letterset"):
                if r.get(k) is not None:
                    accum[t][k].append(r[k])
    print("\n" + "=" * 100)
    print(f"TABLE A0 -- sensitivity of the baselines to PYTHONHASHSEED (n={nseeds} seeds)")
    print("=" * 100)
    print(f"{'task':22s} " + "".join(f"{k:>20s}" for k in
                                     ("majority", "length", "letterset", "length+letterset")))
    art = {"n_seeds": nseeds, "seeds": list(range(nseeds)), "per_task": {}}
    worst = 0.0
    for t in ALL_TASKS:
        line = f"{t:22s} "
        art["per_task"][t] = {}
        for k in ("majority", "length", "letterset", "letterset+q", "length+letterset"):
            v = accum[t][k]
            if v:
                art["per_task"][t][k] = {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                                         "min": float(np.min(v)), "max": float(np.max(v)),
                                         "n": len(v)}
                worst = max(worst, float(np.std(v)))
        for k in ("majority", "length", "letterset", "length+letterset"):
            v = accum[t][k]
            line += f"{np.mean(v):11.3f}+-{np.std(v):<7.3f}" if v else f"{'-':>20s}"
        print(line)
    art["worst_sd"] = worst
    print(f"\nworst per-quantity SD across hash seeds = {worst:.4f}")
    fp = ROOT / "outputs/analysis/hash_seed_sweep.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(art, open(fp, "w"), indent=1)
    print(f"wrote {fp}")


def compute_tables():
    """TABLE A1 quantities only (used by the hash-seed sweep child processes)."""
    by_task = build(SHOTS)
    out = {}
    for task in ALL_TASKS:
        rows, y, g, gfold, folds, classes = prepare(by_task[task])
        r = {"n": len(rows), "n_words": len(set(g)), "n_classes": len(classes)}
        _, r["majority"] = cv_score(rows, y, gfold, folds, make_majority())
        for name, fn in [("length", make_length_only()),
                         ("letterset", make_letterset(task)),
                         ("letterset+q", make_letterset(task, with_query=True)),
                         ("length+letterset", make_letterset(
                             task, with_query=task in QUERY_CHAR_TASKS, with_length=True))]:
            if name == "letterset+q" and task not in QUERY_CHAR_TASKS:
                r[name] = None
                continue
            r[name], _ = cv_score(rows, y, gfold, folds, fn)
        out[task] = r
    return out


def main():
    print("=" * 100)
    print("NUISANCE BASELINES  (held-out-word, folds=%d, min_per=%d, n=%d/task, seed=%d, shots=%d)"
          % (FOLDS, MIN_PER, N_PER_TASK, SEED, SHOTS))
    print("=" * 100)

    by_task = build(SHOTS)
    gate = load_gate(SHOTS)
    models = sorted({m for (m, t) in gate} - GEMMA)
    gemma_models = sorted({m for (m, t) in gate} & GEMMA)

    rows_out = {}
    for task in ALL_TASKS:
        rows = by_task[task]
        rows, y, g, gfold, folds, classes = prepare(rows)
        res = {"n": len(rows), "n_words": len(set(g)), "n_classes": len(classes)}
        acc, maj = cv_score(rows, y, gfold, folds, make_majority())
        res["majority"] = maj
        res["majority_check"] = acc      # must equal majority
        for name, fn in [
            ("length", make_length_only()),
            ("letterset", make_letterset(task)),
            ("letterset+q", make_letterset(task, with_query=True)),
            ("length+letterset", make_letterset(task, with_query=task in QUERY_CHAR_TASKS,
                                                with_length=True)),
        ]:
            if name == "letterset+q" and task not in QUERY_CHAR_TASKS:
                res[name] = None
                continue
            a, _ = cv_score(rows, y, gfold, folds, fn)
            res[name] = a
        rows_out[task] = res

    # ---- cross-check the reconstruction against the recorded majority in the JSONs
    print("\n[VALIDATION] reconstructed majority vs the value recorded in the gate JSONs")
    print(f"{'task':22s} {'recon':>8s} {'json(mean over 14)':>20s} {'max|diff|':>10s}  n_classes recon/json")
    for task in ALL_TASKS:
        js = [gate[(m, task)]["majority"] for m in models if (m, task) in gate]
        jc = [gate[(m, task)]["probe_classes"]["n_classes"] for m in models if (m, task) in gate]
        r = rows_out[task]
        d = max(abs(r["majority"] - v) for v in js)
        print(f"{task:22s} {r['majority']:8.4f} {np.mean(js):20.4f} {d:10.4f}"
              f"   {r['n_classes']}/{sorted(set(jc))}")

    # ---- the nuisance table
    print("\n" + "=" * 100)
    print("TABLE A1 -- nuisance baselines (raw held-out-word accuracy)")
    print("=" * 100)
    hdr = (f"{'task':22s} {'n':>5s} {'#w':>5s} {'#cls':>5s} {'major':>7s} {'length':>7s} "
           f"{'letset':>7s} {'lset+q':>7s} {'len+set':>8s} {'best-nuis':>10s}  flags")
    print(hdr)
    for task in ALL_TASKS:
        r = rows_out[task]
        cand = {k: r[k] for k in ("length", "letterset", "letterset+q", "length+letterset")
                if r.get(k) is not None}
        bn = max(cand.values())
        bnk = max(cand, key=cand.get)
        flags = ",".join(sorted({k for k, v in ORACLE.items() if task in v and k in cand}))
        r["best_nuisance"] = bn
        r["best_nuisance_name"] = bnk
        # strongest NON-oracle nuisance -- the load-bearing number
        nn = {k: v for k, v in cand.items() if task not in ORACLE.get(k, set())}
        r["best_true_nuisance"] = max(nn.values()) if nn else None
        r["best_true_nuisance_name"] = max(nn, key=nn.get) if nn else None
        def s(v):
            return f"{v:7.3f}" if v is not None else f"{'-':>7s}"
        print(f"{task:22s} {r['n']:5d} {r['n_words']:5d} {r['n_classes']:5d} "
              f"{r['majority']:7.3f} {s(r['length'])} {s(r['letterset'])} {s(r['letterset+q'])} "
              f"{r['length+letterset']:8.3f} {bn:8.3f}({bnk[:6]})  "
              f"{'ORACLE:'+flags if flags else 'NUISANCE'}")

    # ---- the headline: model above-majority vs above-nuisance
    print("\n" + "=" * 100)
    print("TABLE A2 -- per task: model representation vs the nuisance floor (shots=3, 14 non-Gemma)")
    print("  above-maj    = knowledge_answer_peak - majority   (what the paper reports)")
    print("  above-nuis   = knowledge_answer_peak - best NON-ORACLE nuisance baseline")
    print("=" * 100)
    print(f"{'task':22s} {'major':>7s} {'nuis':>7s} {'(which)':>10s} "
          f"{'bestRep':>8s} {'best above-maj':>14s} {'best above-nuis':>15s} {'#models>nuis':>13s}")
    survive = {}
    for task in ALL_TASKS:
        r = rows_out[task]
        reps = {m: gate[(m, task)]["knowledge_answer_peak"] for m in models if (m, task) in gate}
        am = {m: gate[(m, task)]["knowledge_above_majority"] for m in models if (m, task) in gate}
        floor = r["best_true_nuisance"]
        bestm = max(reps, key=reps.get)
        an = {m: reps[m] - floor for m in reps}
        nabove = sum(1 for v in an.values() if v > 0)
        survive[task] = dict(floor=floor, floor_name=r["best_true_nuisance_name"],
                             best_model=bestm, best_rep=reps[bestm],
                             best_above_maj=max(am.values()), best_above_nuis=max(an.values()),
                             n_above=nabove, n=len(reps), above_nuis=an, above_maj=am, reps=reps)
        print(f"{task:22s} {r['majority']:7.3f} {floor:7.3f} {r['best_true_nuisance_name'][:10]:>10s} "
              f"{reps[bestm]:8.3f} {max(am.values()):14.3f} {max(an.values()):15.3f} "
              f"{nabove:8d}/{len(reps)}")

    # ---- per-model detail for the count tasks (the contested ones)
    print("\n" + "=" * 100)
    print("TABLE A3 -- per-model detail, compute tasks (shots=3). rep | above-maj | above-nuisance")
    print("=" * 100)
    ctasks = ["counting", "count_distinct", "most_frequent_letter"]
    print(f"{'model':24s}" + "".join(f"{t[:16]:>34s}" for t in ctasks))
    print(f"{'':24s}" + "".join(f"{'rep':>10s}{'a-maj':>11s}{'a-nuis':>13s}" for _ in ctasks))
    for m in models:
        line = f"{m:24s}"
        for t in ctasks:
            s = survive[t]
            line += f"{s['reps'][m]:10.3f}{s['above_maj'][m]:11.3f}{s['above_nuis'][m]:13.3f}"
        print(line)

    # ---- Gemma, reported separately (excluded from everything above)
    print("\n[Gemma-2, reported separately -- EXCLUDED from all analyses above]")
    for m in gemma_models:
        line = f"{m:24s}"
        for t in ctasks:
            g = gate.get((m, t))
            line += f"{g['knowledge_answer_peak']:10.3f}{g['knowledge_above_majority']:11.3f}" \
                    f"{g['knowledge_answer_peak'] - survive[t]['floor']:13.3f}"
        print(line)

    # ---- the confound mechanism
    print("\n" + "=" * 100)
    print("TABLE A4 -- the confound mechanism: mean word length by true answer")
    print("=" * 100)
    for task in ("counting", "count_distinct", "word_length"):
        rows = by_task[task]
        rows, y, g, gfold, folds, classes = prepare(rows)
        print(f"\n{task}:")
        print(f"  {'answer':>8s} {'n':>6s} {'uniq words':>11s} {'mean len':>9s} {'sd':>6s}")
        for c in classes:
            L = [len(r["word"]) for r, a in zip(rows, y) if a == c]
            uw = len({r["word"] for r, a in zip(rows, y) if a == c})
            print(f"  {c:>8s} {len(L):6d} {uw:11d} {np.mean(L):9.2f} {np.std(L):6.2f}")
        # pool sizes over the FULL 6000-word stimulus, not just the sample
        if task == "counting":
            words = s1.load_word_pool(WORDS_FILE)
            pool = defaultdict(list)
            for w in words:
                for ch in set(w):
                    pool[w.count(ch)].append((w, ch))
            print("  full-pool (word,letter) pairs by count: " +
                  ", ".join(f"{c}:{len(pool[c])} ({len({w for w,_ in pool[c]})} words)"
                            for c in (1, 2, 3)))
        if task == "count_distinct":
            words = s1.load_word_pool(WORDS_FILE)
            cd = defaultdict(set)
            for w in words:
                if len(set(w)) <= 9:
                    cd[len(set(w))].add(w)
            print("  full-pool words by n_distinct: " +
                  ", ".join(f"{c}:{len(cd[c])}" for c in sorted(cd)))

    # ---- persist
    outp = ROOT / "outputs" / "analysis" / "nuisance_baselines.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"per_task": rows_out,
                                "survival": {k: {kk: vv for kk, vv in v.items()}
                                             for k, v in survive.items()}},
                               indent=2, default=float))
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-seeds", type=int, default=0,
                    help="also run a PYTHONHASHSEED sensitivity sweep with N seeds")
    ap.add_argument("--emit-json", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.emit_json:
        print(json.dumps(compute_tables(), default=float))
        sys.exit(0)
    # pin set-iteration order so this script's own numbers are reproducible (see module docstring)
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.execve(sys.executable, [sys.executable, __file__] + sys.argv[1:],
                  dict(os.environ, PYTHONHASHSEED="0"))
    main()
    if a.hash_seeds:
        hash_seed_sweep(a.hash_seeds)
