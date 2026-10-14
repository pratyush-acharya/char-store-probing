#!/usr/bin/env python3
"""Unified leakage-safe grid runner (Tier B+C efficiency).

Loads the model ONCE per invocation and loops over shots x sets in-process; for each cell it does
ONE forward pass (all layers) shared by the gate, the unembedding-alignment, and (at --gen-shots)
the random-vs-held-out-word drop + free-generation robustness. Reuses the exact metric functions
from run_stage1 / run_alignment / run_grouped_probe, and writes the SAME output files
(outputs/bigstim_*), so downstream analyzers are unchanged. `--skip-existing` skips per output file.

This replaces N separate `uv run` processes (one model load + one forward each) with one load and
one forward per cell -- the dominant cost cut for the big-model grid.

Usage:
  uv run python scripts/run_grid.py --model Qwen/Qwen3-8B --shots 0,1,3,5 --sets orig,cvs \
      --words-file src/stimulus_words.txt --batch-size 48 --skip-existing
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
import run_alignment as al
import run_grouped_probe as gp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _tag(model):
    return f"_{s1.model_tag(model)}"


def _tagged(kind, model, taskset, shots, out_tag):
    """Control runs write to their OWN dir (outputs/bigstim_<tag>/) so they can never clobber the
    published grid and --skip-existing stays meaningful per control."""
    return f"outputs/bigstim_{out_tag}/{kind}_{taskset}{_tag(model)}_s{shots}.json"


def gate_path(model, taskset, shots, seedvar=False, seed=None, out_tag=None):
    if out_tag:
        return _tagged("gate", model, taskset, shots, out_tag)
    if seedvar:
        return f"outputs/bigstim_seedvar/gate_{taskset}{_tag(model)}_s{shots}_seed{seed}.json"
    base = "bigstim_stage1_gate_grouped" + ("" if taskset == "orig" else f"_{taskset}")
    return f"outputs/{base}/results{_tag(model)}_s{shots}.json"


def align_path(model, taskset, shots, seedvar=False, seed=None, out_tag=None):
    if out_tag:
        return _tagged("align", model, taskset, shots, out_tag)
    if seedvar:
        return f"outputs/bigstim_seedvar/align_{taskset}{_tag(model)}_s{shots}_seed{seed}.json"
    return f"outputs/bigstim_robustness/aligngrouped_{taskset}{_tag(model)}_s{shots}.json"


def drop_path(model, taskset, shots, seedvar=False, seed=None, out_tag=None):
    if out_tag:
        return _tagged("drop", model, taskset, shots, out_tag)
    if seedvar:
        return f"outputs/bigstim_seedvar/drop_{taskset}{_tag(model)}_s{shots}_seed{seed}.json"
    return f"outputs/bigstim_robustness/grouped_{taskset}{_tag(model)}_s{shots}.json"


def _write(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", required=True)
    ap.add_argument("--shots", default="0,1,3,5", help="comma-separated shot counts")
    ap.add_argument("--sets", default="orig,cvs", help="comma-separated task sets")
    ap.add_argument("--n", type=int, default=800, help="examples per task")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--words-file", default="src/stimulus_words.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--probe-stride", type=int, default=4, help="layer stride for the probe peak")
    ap.add_argument("--group-folds", type=int, default=3, help="held-out-word CV folds")
    ap.add_argument("--gen-shots", type=int, default=3,
                    help="compute the drop + free-generation robustness only at this shot count")
    ap.add_argument("--no-remote-code", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--seedvar", action="store_true",
                    help="seed-variance mode: write seed-tagged outputs to outputs/bigstim_seedvar/ "
                         "(so multiple --seed runs don't overwrite each other or the main grid)")
    # --- CONTROL EXPERIMENTS (all off by default -> published behaviour unchanged) ---
    ap.add_argument("--stages", default="gate,align,drop",
                    help="CONTROL: which stages to compute (subset of gate,align,drop). Running a "
                         "single stage skips the others' cost entirely; 'align' alone also keeps "
                         "only the final layer's hidden states.")
    ap.add_argument("--out-tag", default=None, metavar="TAG",
                    help="CONTROL: write to outputs/bigstim_TAG/ instead of the published dirs")
    ap.add_argument("--rank-control", type=int, default=0, metavar="N",
                    help="CONTROL (P2): N random rank-matched subspaces + top-r_out PCs in the "
                         "alignment stage. No extra forward passes.")
    ap.add_argument("--probe-layers", type=int, default=0, metavar="N",
                    help="CONTROL (P3): probe a FIXED NUMBER of evenly-spaced layers per model "
                         "instead of an absolute --probe-stride")
    ap.add_argument("--layer-select", default="max", choices=["max", "heldout"],
                    help="CONTROL (P3): 'heldout' also reports a fold-held-out layer choice")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="CONTROL (P4): word-level bootstrap resamples for 95%% CIs")
    ap.add_argument("--cold-start", action="store_true",
                    help="CONTROL (P6): disable the warm-start across CV folds")
    args = ap.parse_args()

    shots_list = [int(s) for s in str(args.shots).split(",") if s != ""]
    sets_list = [s for s in args.sets.split(",") if s]

    config = ExperimentConfig()
    config.model.model_name = args.model
    apply_env_overrides(config)
    if args.no_remote_code:
        config.model.trust_remote_code = False

    sv, sd, ot = args.seedvar, args.seed, args.out_tag
    stages = {s.strip() for s in args.stages.split(",") if s.strip()}
    unknown = stages - {"gate", "align", "drop"}
    if unknown:
        raise SystemExit(f"unknown --stages: {sorted(unknown)}")

    def cell_paths(ts, sh):
        need = []
        if "gate" in stages:
            need.append(gate_path(args.model, ts, sh, sv, sd, ot))
        if "align" in stages:
            need.append(align_path(args.model, ts, sh, sv, sd, ot))
        if "drop" in stages and sh == args.gen_shots:
            need.append(drop_path(args.model, ts, sh, sv, sd, ot))
        return need

    if args.skip_existing and all(Path(p).exists()
                                  for sh in shots_list for ts in sets_list for p in cell_paths(ts, sh)):
        logger.info("skip (all outputs exist): %s", args.model)
        return

    logger.info("Loading %s on %s (ONCE)", config.model.model_name, args.device)
    model = CausalLM(config.model).to(args.device)
    model.eval()
    tokenizer = load_tokenizer(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    words = s1.load_word_pool(args.words_file)
    logger.info("word pool: %d words (%s)", len(words), args.words_file)
    n_layers = model.base_model.config.num_hidden_layers
    if args.probe_layers:                       # CONTROL P3: fixed layer COUNT, depth-comparable
        probe_layers = s1.even_layers(n_layers + 1, args.probe_layers)
    else:
        probe_layers = list(range(0, n_layers + 1, args.probe_stride))
    # align-only runs need just the final layer -> far less host RAM / device->host copying
    keep_layers = [n_layers] if stages == {"align"} else None
    logger.info("stages=%s probe_layers=%s out_tag=%s", sorted(stages), probe_layers, ot)

    for sh in shots_list:
        for ts in sets_list:
            gpth = gate_path(args.model, ts, sh, sv, sd, ot)
            apth = align_path(args.model, ts, sh, sv, sd, ot)
            do_drop = ("drop" in stages) and (sh == args.gen_shots)
            dpth = drop_path(args.model, ts, sh, sv, sd, ot)
            need_gate = ("gate" in stages) and not (args.skip_existing and Path(gpth).exists())
            need_align = ("align" in stages) and not (args.skip_existing and Path(apth).exists())
            need_drop = do_drop and not (args.skip_existing and Path(dpth).exists())
            if not (need_gate or need_align or need_drop):
                logger.info("skip (exists) s%d %s", sh, ts)
                continue

            task_list = s1.TASK_SETS[ts]
            examples = s1.build_examples(words, args.n, args.seed, shots=sh,
                                         tokenizer=tokenizer, tasks=task_list)
            ex_words = np.array([e["word"] for e in examples])
            # ONE forward pass over all layers; shared by gate + alignment + drop
            data = s1.extract(model, tokenizer, examples, args.device,
                              batch_size=args.batch_size, keep_layers=keep_layers, want_word=False)
            meta = {"model": args.model, "shots": sh, "set": ts, "seed": args.seed, "grouped": True,
                    "words_file": args.words_file, "n_words": len(words), "n_per_task": args.n,
                    "controls": {"probe_layers": args.probe_layers or None,
                                 "layer_select": args.layer_select,
                                 "bootstrap": args.bootstrap or None,
                                 "rank_control": args.rank_control or None,
                                 "cold_start": bool(args.cold_start)}}

            if need_gate:
                gm = s1.gate_metrics(model, tokenizer, data, task_list, args.device, ex_words,
                                     probe_stride=args.probe_stride, probe_folds=3, group_probe=True,
                                     group_folds=args.group_folds, skip_word_probe=True,
                                     layers=(probe_layers if args.probe_layers else None),
                                     layer_select=args.layer_select, n_boot=args.bootstrap,
                                     warm_start=not args.cold_start)
                _write(gpth, {**gm, "injection_sanity": {},
                              "meta": {**meta, "n_layers": n_layers + 1, "read_layer": 22,
                                       "smoke": False}})
            if need_align:
                am = al.alignment_metrics(model, tokenizer, data, task_list, args.device, True,
                                          ex_words, folds=args.group_folds,
                                          rank_control=args.rank_control, rank_seed=args.seed)
                _write(apth, {"gap": am, "meta": {**meta, "n_layers": n_layers}})
            if need_drop:
                dm = gp.grouped_drop_metrics(data, task_list, ex_words, args.device,
                                             probe_layers, folds=args.group_folds)
                for t in list(dm):
                    fg = gp.free_gen_check(model, tokenizer, examples, args.device, t)
                    if fg:
                        dm[t]["free_gen"] = fg
                _write(dpth, {"gap": dm, "meta": {**meta, "folds": args.group_folds}})
            logger.info("done s%d %s (gate=%s align=%s drop=%s)", sh, ts,
                        need_gate, need_align, need_drop)

    logger.info("grid complete for %s", args.model)


if __name__ == "__main__":
    main()
