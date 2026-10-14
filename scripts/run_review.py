#!/usr/bin/env python3
"""Reviewer follow-ups R1 and R2 for the paper (GPU). Pre-registered before running.

R1  TOKEN SPLIT. The answer position under the question exceeds the word-level store in larger
    models. Is that excess only cross-token integration? Re-evaluate the SAME held-out-word linear
    probes (fit on all examples, held-out layer selection) separately on examples whose query word
    is a single token vs several tokens, at three positions: answer@question, word@neutral (last
    token) and word@neutral (span mean). For a single-token word the two word reads coincide.
R2  MLP PROBE. Is the excess already present NONLINEARLY at the word tokens, with late layers merely
    linearizing it? Fit a 2-layer MLP probe (same folds, held-out layer selection) at word@neutral
    (last token, span mean) and at answer@question, and compare with the linear probe.

Tasks: the pure-word tasks first_letter, last_letter (set orig), most_frequent_letter, word_length
(set cvs). Examples, prompts, folds and layers are identical to run_wordpos.py.

Usage:
  uv run python scripts/run_review.py --model Qwen/Qwen3-8B --shots 0,3 --mlp-shots 0 \
      --words-file src/stimulus_words.txt --skip-existing
Output: outputs/v4/review/review_{set}_{model}_s{shots}.json
"""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import ExperimentConfig, apply_env_overrides
from src.model import CausalLM, load_tokenizer
import run_stage1 as s1
import run_noquery as nq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TASKS = {"orig": ["first_letter", "last_letter"], "cvs": ["most_frequent_letter", "word_length"]}
MLP_HIDDEN, MLP_STEPS, MLP_LR, MLP_WD, MLP_DROPOUT = 256, 300, 1e-3, 1e-2, 0.1


def out_path(out_dir, model, taskset, shots):
    return f"{out_dir}/review_{taskset}_{s1.model_tag(model)}_s{shots}.json"


# ----------------------------------------------------------------------------- R1: linear, split
def linear_split(H_by_layer, y, groups, device, layers, subsets, folds=3):
    """Linear held-out-word probe with held-out layer selection; per-example held-out correctness
    at each fold's selected layer, then accuracy (and majority) within each subset mask."""
    per_layer = {}
    for L in layers:
        r = s1.probe_acc_grouped(H_by_layer[L], y, groups, device, folds=folds, warm_start=False,
                                 return_preds=True)
        if r:
            per_layer[L] = r
    if not per_layer:
        return None
    any_r = next(iter(per_layer.values()))
    kept, fold_of = any_r["kept_mask"], any_r["ex_fold"]
    fold_ids = sorted(set(any_r["fold_ids"]))
    correct = np.full(int(kept.sum()), np.nan)
    majority = np.full(int(kept.sum()), np.nan)
    chosen = {}
    for f in fold_ids:
        best_L = max(per_layer, key=lambda L: np.mean([a for g, a in zip(per_layer[L]["fold_ids"],
                                                                         per_layer[L]["fold_accs"])
                                                       if g != f] or [-1]))
        chosen[int(f)] = int(best_L)
        m = fold_of == f
        correct[m] = per_layer[best_L]["ex_correct"][m]
        majority[m] = per_layer[best_L]["ex_majority"][m]
    out = {"all": {"acc": float(np.nanmean(correct)), "majority": float(np.nanmean(majority)),
                   "n": int(np.sum(~np.isnan(correct)))}, "heldout_layers": chosen}
    for name, mask in subsets.items():
        mk = np.asarray(mask)[kept]
        ok = mk & ~np.isnan(correct)
        if ok.sum() >= 20:
            out[name] = {"acc": float(correct[ok].mean()), "majority": float(majority[ok].mean()),
                         "n": int(ok.sum())}
    return out


# ----------------------------------------------------------------------------- R2: MLP probe
def _mlp_fit_eval(Xtr, ytr, Xte, yte, k, device, seed):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    net = torch.nn.Sequential(torch.nn.Dropout(MLP_DROPOUT), torch.nn.Linear(Xtr.shape[1], MLP_HIDDEN),
                              torch.nn.ReLU(), torch.nn.Linear(MLP_HIDDEN, k)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=MLP_LR, weight_decay=MLP_WD)
    net.train()
    for _ in range(MLP_STEPS):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(Xtr), ytr).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        return float((net(Xte).argmax(1) == yte).float().mean().item())


def mlp_probe(H_by_layer, y, groups, device, layers, folds=3, min_per=5):
    """2-layer MLP, same class filter, grouped folds and held-out layer selection as the linear
    probe (s1.probe_acc_grouped). Returns {acc_heldout_sel, by_layer, heldout_layers}."""
    classes, counts = np.unique(y, return_counts=True)
    keep = classes[counts >= min_per]
    if len(keep) < 2:
        return None
    mask = np.isin(y, keep)
    remap = {int(c): i for i, c in enumerate(keep)}
    yi = torch.tensor([remap[int(v)] for v in y[mask]], device=device)
    g = np.asarray(groups)[mask]
    uniq = np.array(sorted(set(g.tolist())))
    order = np.random.RandomState(0).permutation(len(uniq))
    fold_of = {w: i % folds for i, w in enumerate(uniq[order])}
    gf = np.array([fold_of[w] for w in g])
    tab = {}
    for L in layers:
        X = torch.tensor(H_by_layer[L][mask], dtype=torch.float32, device=device)
        accs = {}
        for f in range(folds):
            te = torch.tensor(np.where(gf == f)[0], device=device)
            tr = torch.tensor(np.where(gf != f)[0], device=device)
            accs[f] = _mlp_fit_eval(X[tr], yi[tr], X[te], yi[te], len(keep), device, seed=f)
        tab[int(L)] = accs
    sel, chosen = [], {}
    for f in range(folds):
        best_L = max(tab, key=lambda L: np.mean([a for g2, a in tab[L].items() if g2 != f]))
        chosen[f] = best_L
        sel.append(tab[best_L][f])
    return {"acc_heldout_sel": round(float(np.mean(sel)), 4), "heldout_layers": chosen,
            "by_layer": {L: round(float(np.mean(list(a.values()))), 4) for L, a in tab.items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", required=True)
    ap.add_argument("--shots", default="0,3")
    ap.add_argument("--mlp-shots", default="0", help="shot counts at which to fit the MLP probe")
    ap.add_argument("--sets", default="orig,cvs")
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--words-file", default="src/stimulus_words.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--probe-layers", type=int, default=12)
    ap.add_argument("--group-folds", type=int, default=3)
    ap.add_argument("--out-dir", default="outputs/v4/review")
    ap.add_argument("--no-remote-code", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    shots_list = [int(x) for x in args.shots.split(",") if x != ""]
    mlp_shots = {int(x) for x in args.mlp_shots.split(",") if x != ""}
    sets_list = [x for x in args.sets.split(",") if x]
    todo = [(sh, ts) for sh in shots_list for ts in sets_list
            if not (args.skip_existing and Path(out_path(args.out_dir, args.model, ts, sh)).exists())]
    if not todo:
        logger.info("skip (all outputs exist): %s", args.model)
        return

    config = ExperimentConfig()
    config.model.model_name = args.model
    apply_env_overrides(config)
    if args.no_remote_code:
        config.model.trust_remote_code = False
    model = CausalLM(config.model).to(args.device)
    model.eval()
    tokenizer = load_tokenizer(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    words = s1.load_word_pool(args.words_file)
    layers = s1.even_layers(model.base_model.config.num_hidden_layers + 1, args.probe_layers)

    for sh, ts in todo:
        task_list = TASKS[ts]
        examples = s1.build_examples(words, args.n, args.seed, shots=sh, tokenizer=tokenizer,
                                     tasks=s1.TASK_SETS[ts])          # same RNG stream as the grid
        examples = [e for e in examples if e["task"] in task_list]
        ex_words = np.array([e["word"] for e in examples])
        neutral = nq.build_carrier_examples(examples, words, "neutral", sh, args.seed)
        dq = s1.extract(model, tokenizer, examples, args.device, batch_size=args.batch_size,
                        keep_layers=layers, want_word=True, word_mean=False)
        dn = s1.extract(model, tokenizer, neutral, args.device, batch_size=args.batch_size,
                        keep_layers=layers, want_word=True, word_mean=True)
        # token count of the query word, measured in the neutral sentence (same tokenizer, same word)
        ntok = dn["word_ntok"]
        res = {}
        for t in task_list:
            m = dq["tasks"] == t
            y, g = dq["target_ids"][m], ex_words[m]
            subsets = {"single_token": ntok[m] == 1, "multi_token": ntok[m] > 1}
            positions = {"answer_question": {L: dq["ans_h"][L][m] for L in layers},
                         "word_neutral_last": {L: dn["word_h"][L][m] for L in layers},
                         "word_neutral_mean": {L: dn["word_mean_h"][L][m] for L in layers}}
            row = {"n_single": int(subsets["single_token"].sum()),
                   "n_multi": int(subsets["multi_token"].sum()),
                   "linear": {p: linear_split(H, y, g, args.device, layers, subsets, args.group_folds)
                              for p, H in positions.items()}}
            if sh in mlp_shots:
                row["mlp"] = {p: mlp_probe(H, y, g, args.device, layers, args.group_folds)
                              for p, H in positions.items()}
            res[t] = row
            lin = row["linear"]
            logger.info("s%d %-20s single=%d multi=%d  ans@q single %.3f multi %.3f | word_last single %.3f",
                        sh, t, row["n_single"], row["n_multi"],
                        (lin["answer_question"].get("single_token") or {}).get("acc", np.nan),
                        (lin["answer_question"].get("multi_token") or {}).get("acc", np.nan),
                        (lin["word_neutral_last"].get("single_token") or {}).get("acc", np.nan))
        p = Path(out_path(args.out_dir, args.model, ts, sh))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"results": res, "meta": {
            "model": args.model, "shots": sh, "set": ts, "tasks": task_list, "layers": layers,
            "n_hidden": model.base_model.config.num_hidden_layers + 1, "seed": args.seed,
            "n_per_task": args.n, "mlp": {"hidden": MLP_HIDDEN, "steps": MLP_STEPS, "lr": MLP_LR,
                                          "weight_decay": MLP_WD, "dropout": MLP_DROPOUT,
                                          "fit_at_shots": sorted(mlp_shots)},
            "estimator": "held-out-word GroupKFold, cold start, held-out layer selection"}}, indent=1))
        logger.info("saved -> %s", p)


if __name__ == "__main__":
    main()
