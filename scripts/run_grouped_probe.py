#!/usr/bin/env python3
"""Robustness check for the "representation contains the answer" claim: does the answer probe
survive HELD-OUT WORDS, or is it a memorized word->answer lookup over the ~100-word pool?

The main gate (scripts/run_stage1.py, probe_acc_gpu) uses repeated random 70/30 splits, so the
same query word can appear in train and test. For tasks whose answer is a deterministic function
of the word (e.g. first/last-letter), a high-capacity linear probe could in principle memorize a
word->answer table rather than read a general character feature. Here we re-probe with
GroupKFold BY QUERY WORD (no word in both train and test) and report the two side by side:

  acc_random  : peak-over-layers CV accuracy, random 70/30 splits (== the gate's number)
  acc_grouped : peak-over-layers CV accuracy, held-out-WORD folds (leakage-safe)
  drop        : acc_random - acc_grouped   (large -> the gate number was partly lexical lookup)

We also run a small FREE-GENERATION sanity check (--gen-check): for a sample of queries, greedily
generate a few tokens with NO answer-space restriction and score whether the target character/字
appears -- a check that the "answer-space readout" story is not an artifact of restricted argmax.

Usage:
  uv run python scripts/run_grouped_probe.py --model Qwen/Qwen3-0.6B --shots 3 [--device mps]
  uv run python scripts/run_grouped_probe.py --model Qwen/Qwen3-8B --set orig --gen-check
Output: outputs/robustness/grouped_{model}_{set}_s{shots}.json
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@torch.no_grad()
def free_gen_check(model, tokenizer, examples, device, task, n=60, max_new=6):
    """Unrestricted greedy free generation: does the target char/digit appear in the first few
    generated tokens? Compares against the answer-space (restricted argmax) readout."""
    sub = [e for e in examples if e["task"] == task][:n]
    if not sub:
        return None
    hit = 0
    for e in sub:
        enc = tokenizer(e["prompt"], return_tensors="pt").to(device)
        out = model.base_model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
        gen = tokenizer.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
        tgt = str(e["answer"]).strip().lower()
        # first non-space run of the generation, and whether the target token appears verbatim
        g = gen.strip().lower()
        first = g.split()[0] if g.split() else ""
        if tgt and (first.startswith(tgt) or (f" {tgt}" in f" {g}"[:12])):
            hit += 1
    return {"free_gen_acc": round(hit / len(sub), 4), "n": len(sub), "max_new": max_new}


def grouped_drop_metrics(data, task_list, ex_words, device, probe_layers, folds=3):
    """Per task: peak-over-layers random-split vs held-out-word probe accuracy + the drop and the
    above-majority margin, from an already-extracted `data`. Shared by main() and run_grid.py.
    (free_gen is added by the caller.) Returns {task: entry}."""
    tasks, targets = data["tasks"], data["target_ids"]
    ex_words = np.asarray(ex_words)
    results = {}
    for t in task_list:
        m = tasks == t
        if m.sum() == 0:
            continue
        y, gwords = targets[m], ex_words[m]
        best_rand = best_grp = None
        for L in probe_layers:
            if L not in data["ans_h"]:
                continue
            H = data["ans_h"][L][m]
            r = s1.probe_acc_gpu(H, y, device, folds=3)
            g = s1.probe_acc_grouped(H, y, gwords, device, folds=folds)
            if r and (best_rand is None or r["acc"] > best_rand["acc"]):
                best_rand = {**r, "layer": L}
            if g and (best_grp is None or g["acc"] > best_grp["acc"]):
                best_grp = {**g, "layer": L}
        if not best_rand or not best_grp:
            continue
        chance = best_rand["chance"]
        results[t] = {
            "chance": round(chance, 3),
            "majority": round(best_grp.get("majority", chance), 4),
            "acc_random": round(best_rand["acc"], 4), "layer_random": best_rand["layer"],
            "acc_grouped": round(best_grp["acc"], 4), "layer_grouped": best_grp["layer"],
            "drop": round(best_rand["acc"] - best_grp["acc"], 4),
            "above_majority": round(best_grp["acc"] - best_grp.get("majority", chance), 4),
            "n_words": best_grp["n_words"], "grouped_folds": best_grp["folds"],
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--model", default=None)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--set", dest="taskset", default="orig", choices=list(s1.TASK_SETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--probe-stride", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5, help="grouped (held-out-word) folds")
    ap.add_argument("--gen-check", action="store_true", help="also run the free-generation sanity check")
    ap.add_argument("--no-remote-code", action="store_true")
    ap.add_argument("--words-file", default=None,
                    help="large stimulus word list (src/stimulus_words.txt) -> leakage-safe pool; "
                         "outputs go to bigstim_robustness/. Omit for the legacy ~100-word pool.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip if the output file already exists (cheap restarts)")
    ap.add_argument("--out", default="outputs/robustness/grouped.json")
    args = ap.parse_args()

    task_list = s1.TASK_SETS[args.taskset]
    config = ExperimentConfig()
    mt = f"_{s1.model_tag(args.model)}" if args.model else ""
    if args.model:
        config.model.model_name = args.model
    if args.words_file:
        args.out = args.out.replace("outputs/robustness/", "outputs/bigstim_robustness/")
    args.out = args.out.replace(".json", f"_{args.taskset}{mt}_s{args.shots}.json")
    if args.skip_existing and Path(args.out).exists():
        logger.info("skip (exists): %s", args.out); return
    apply_env_overrides(config)
    if args.no_remote_code:
        config.model.trust_remote_code = False

    logger.info("Loading %s on %s", config.model.model_name, args.device)
    model = CausalLM(config.model).to(args.device)
    model.eval()
    tokenizer = load_tokenizer(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    words = s1.load_word_pool(args.words_file)
    logger.info("word pool: %d words (%s)", len(words), args.words_file or "built-in ~100-word pool")
    examples = s1.build_examples(words, args.n, args.seed, shots=args.shots,
                                 tokenizer=tokenizer, tasks=task_list)
    ex_words = np.array([e["word"] for e in examples])

    n_layers = model.base_model.config.num_hidden_layers
    probe_layers = list(range(0, n_layers + 1, args.probe_stride))
    data = s1.extract(model, tokenizer, examples, args.device,
                      batch_size=args.batch_size, keep_layers=probe_layers, want_word=False)

    results = grouped_drop_metrics(data, task_list, ex_words, args.device, probe_layers, args.folds)
    if args.gen_check:
        for t in list(results):
            fg = free_gen_check(model, tokenizer, examples, args.device, t)
            if fg:
                results[t]["free_gen"] = fg
    for t, entry in results.items():
        logger.info("%-16s acc_random=%.3f@L%d  acc_grouped=%.3f@L%d  drop=%+.3f  (chance %.3f, %d words)",
                    t, entry["acc_random"], entry["layer_random"], entry["acc_grouped"],
                    entry["layer_grouped"], entry["drop"], entry["chance"], entry["n_words"])

    out = {"gap": results, "meta": {"model": config.model.model_name, "shots": args.shots,
                                    "set": args.taskset, "seed": args.seed, "folds": args.folds,
                                    "words_file": args.words_file, "n_words": len(words)}}
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 84)
    print("HELD-OUT-WORD PROBE ROBUSTNESS  (random-split vs word-grouped; drop = leakage)")
    print("=" * 84)
    print(f"{'task':16} {'chance':>7} {'random':>8} {'grouped':>8} {'drop':>7}  {'free_gen':>8}")
    for t in task_list:
        if t not in results:
            continue
        r = results[t]
        fg = r.get("free_gen", {}).get("free_gen_acc")
        print(f"{t:16} {r['chance']:7.3f} {r['acc_random']:8.3f} {r['acc_grouped']:8.3f} "
              f"{r['drop']:+7.3f}  {('%.3f' % fg) if fg is not None else '--':>8}")
    print(f"\nSaved -> {p}")


if __name__ == "__main__":
    main()
