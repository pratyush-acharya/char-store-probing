"""Within-family scaling of representation, readout, and the readout gap (LOCAL, no GPU).

The paper's contribution (2) -- "the KPG is a readout gap that shrinks with scale" -- is
supported by a POOLED cross-family correlation. Pooled correlations across architecture families
confound scale with family. This script tabulates, at shots=3, per family with >=3 sizes:

    representation  = gap[task]["knowledge_answer_peak"]        (leakage-safe held-out-word probe)
    readout         = gap[task]["prediction_acc"]               (answer-space argmax)
    gap             = representation - readout                  (== gap_answer_vs_pred)

so the WITHIN-family trajectory is visible, and computes r(log10 params, .) across the 14
non-Gemma models per task. Gemma-2 is excluded everywhere (padding_side measurement failure)
and reported separately at the end.

Run:  uv run python scripts/analyze_within_family.py
      uv run python scripts/analyze_within_family.py --latex
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# (family, params_B). Params are TOTAL params, matching make_paper_figures.SPEC, with an
# additional ACTIVE-param reading for OLMoE (1B active / 7B total) -- see --olmoe-active.
SPEC = {
    "Qwen3_0_6B": ("Qwen3", 0.6), "Qwen3_1_7B": ("Qwen3", 1.7), "Qwen3_4B": ("Qwen3", 4),
    "Qwen3_8B": ("Qwen3", 8), "Llama_3_2_1B": ("Llama", 1), "Llama_3_2_3B": ("Llama", 3),
    "Llama_3_1_8B": ("Llama", 8), "gemma_2_2b": ("Gemma2", 2), "gemma_2_9b": ("Gemma2", 9),
    "OLMo_2_0425_1B": ("OLMo2", 1), "OLMo_2_1124_7B": ("OLMo2", 7), "OLMo_2_1124_13B": ("OLMo2", 13),
    "Olmo_Hybrid_7B": ("OLMoHybrid", 7), "OLMoE_1B_7B_0924": ("OLMoE", 7),
    "Phi_4_mini_instruct": ("Phi", 3.8), "phi_4": ("Phi", 14),
}
GEMMA = {"gemma_2_2b", "gemma_2_9b"}
POST_TRAINED = {"Qwen3_0_6B", "Qwen3_1_7B", "Qwen3_4B", "Qwen3_8B",
                "Phi_4_mini_instruct", "phi_4"}
FAMILIES_3PLUS = ["Qwen3", "Llama", "OLMo2"]      # >=3 sizes
FAMILIES_2 = ["Phi"]                               # 2 sizes, reported for completeness
TASKS = ["first_letter", "last_letter", "contains_letter", "extraction", "counting",
         "most_frequent_letter", "letter_before", "word_length", "count_distinct"]
COMPUTE_TASKS = ["counting", "count_distinct", "most_frequent_letter"]
RETRIEVAL_TASKS = ["first_letter", "last_letter", "extraction", "letter_before"]


def load_gate(shots):
    """{(model, task): gap-dict} from the 6000-word grouped gate dirs."""
    out = {}
    for sub in ("bigstim_stage1_gate_grouped", "bigstim_stage1_gate_grouped_cvs"):
        for fp in glob.glob(str(ROOT / "outputs" / sub / f"results_*_s{shots}.json")):
            model = re.match(r".*results_(.+)_s\d\.json", fp).group(1)
            if model not in SPEC:
                continue
            for t, g in json.load(open(fp))["gap"].items():
                out[(model, t)] = g
    return out


def cells(gate, model, task):
    g = gate.get((model, task))
    if g is None:
        return None
    rep, ro = g["knowledge_answer_peak"], g["prediction_acc"]
    return dict(rep=rep, readout=ro, gap=rep - ro, above_maj=g["knowledge_above_majority"],
                majority=g["majority"])


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def r_pvalue(r, n):
    """Two-sided p for a Pearson r with n observations (t = r*sqrt(n-2)/sqrt(1-r^2))."""
    if not np.isfinite(r) or n <= 2 or abs(r) >= 1:
        return float("nan")
    from scipy import stats
    t = r * np.sqrt(n - 2) / np.sqrt(1 - r * r)
    return float(2 * stats.t.sf(abs(t), n - 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--latex", action="store_true", help="also emit a LaTeX table for compute tasks")
    args = ap.parse_args()
    gate = load_gate(args.shots)
    models = [m for m in SPEC if m not in GEMMA and any((m, t) in gate for t in TASKS)]
    by_fam = defaultdict(list)
    for m in models:
        by_fam[SPEC[m][0]].append(m)
    for f in by_fam:
        by_fam[f].sort(key=lambda m: SPEC[m][1])

    print("=" * 108)
    print(f"WITHIN-FAMILY SCALING at shots={args.shots}   (14 non-Gemma models; Gemma-2 excluded)")
    print("  rep = knowledge_answer_peak (held-out-word probe) | readout = prediction_acc | gap = rep - readout")
    print("=" * 108)

    # ---------------- B1: per-family trajectories, all tasks
    for fam in FAMILIES_3PLUS + FAMILIES_2:
        ms = by_fam[fam]
        print(f"\n--- {fam}  ({len(ms)} sizes: "
              f"{', '.join(f'{SPEC[m][1]}B' + ('*' if m in POST_TRAINED else '') for m in ms)})"
              f"   *=post-trained")
        print(f"{'task':22s}" + "".join(f"{SPEC[m][1]:>8}B" for m in ms) +
              "   | metric   |  first->last")
        for task in TASKS:
            for metric in ("rep", "readout", "gap"):
                vals = []
                for m in ms:
                    c = cells(gate, m, task)
                    vals.append(c[metric] if c else float("nan"))
                delta = vals[-1] - vals[0]
                arrow = "shrinks" if (metric == "gap" and delta < -0.02) else \
                        ("GROWS" if (metric == "gap" and delta > 0.02) else
                         ("flat" if metric == "gap" else ""))
                name = task if metric == "rep" else ""
                print(f"{name:22s}" + "".join(f"{v:9.3f}" for v in vals) +
                      f"   | {metric:8s} | {delta:+7.3f} {arrow}")

    # ---------------- B2: focused compute-gap table (the contested claim)
    print("\n" + "=" * 108)
    print("TABLE B1 -- WITHIN-FAMILY readout gap (rep - readout) for the COMPUTE tasks, shots=%d"
          % args.shots)
    print("=" * 108)
    print(f"{'family':12s} {'task':22s} " + "  ".join(f"{'':>10s}" for _ in range(4)) + " verdict")
    for fam in FAMILIES_3PLUS + FAMILIES_2:
        ms = by_fam[fam]
        for task in COMPUTE_TASKS:
            traj = [cells(gate, m, task)["gap"] for m in ms]
            d = traj[-1] - traj[0]
            verdict = "SHRINKS" if d < -0.02 else ("GROWS" if d > 0.02 else "flat")
            s = " -> ".join(f"{v:+.3f}" for v in traj)
            print(f"{fam:12s} {task:22s} {s:52s} delta={d:+.3f}  {verdict}")
        print()

    # also for retrieval, so the contrast is explicit
    print("TABLE B2 -- same, RETRIEVAL tasks (the claim that does survive)")
    for fam in FAMILIES_3PLUS + FAMILIES_2:
        ms = by_fam[fam]
        for task in RETRIEVAL_TASKS:
            traj = [cells(gate, m, task)["gap"] for m in ms]
            d = traj[-1] - traj[0]
            verdict = "SHRINKS" if d < -0.02 else ("GROWS" if d > 0.02 else "flat")
            s = " -> ".join(f"{v:+.3f}" for v in traj)
            print(f"{fam:12s} {task:22s} {s:52s} delta={d:+.3f}  {verdict}")
        print()

    # ---------------- B3: pooled correlations with n and p
    print("=" * 108)
    print("TABLE B3 -- pooled correlation with log10(params) across the 14 non-Gemma models")
    print("=" * 108)
    lp = [np.log10(SPEC[m][1]) for m in models]
    print(f"{'task':22s} {'n':>3s} {'r(above-maj)':>13s} {'p':>8s} {'r(gap)':>9s} {'p':>8s} "
          f"{'r(rep)':>9s} {'r(readout)':>11s}")
    corr = {}
    for task in TASKS:
        am = [cells(gate, m, task)["above_maj"] for m in models]
        gp = [cells(gate, m, task)["gap"] for m in models]
        rp = [cells(gate, m, task)["rep"] for m in models]
        ro = [cells(gate, m, task)["readout"] for m in models]
        r1, r2 = pearson(lp, am), pearson(lp, gp)
        corr[task] = dict(n=len(models), r_above_maj=r1, r_gap=r2,
                          r_rep=pearson(lp, rp), r_readout=pearson(lp, ro))
        print(f"{task:22s} {len(models):3d} {r1:13.3f} {r_pvalue(r1, len(models)):8.4f} "
              f"{r2:9.3f} {r_pvalue(r2, len(models)):8.4f} "
              f"{pearson(lp, rp):9.3f} {pearson(lp, ro):11.3f}")

    # sensitivity: OLMoE plotted at 1B ACTIVE params instead of 7B total
    print("\n[sensitivity] same correlations with OLMoE at 1B ACTIVE params:")
    lp2 = [np.log10(1.0 if m == "OLMoE_1B_7B_0924" else SPEC[m][1]) for m in models]
    print(f"{'task':22s} {'r(above-maj)':>13s} {'r(gap)':>9s}")
    for task in TASKS:
        am = [cells(gate, m, task)["above_maj"] for m in models]
        gp = [cells(gate, m, task)["gap"] for m in models]
        print(f"{task:22s} {pearson(lp2, am):13.3f} {pearson(lp2, gp):9.3f}")

    # sensitivity: base models only (drop the 6 post-trained)
    base = [m for m in models if m not in POST_TRAINED]
    lp3 = [np.log10(SPEC[m][1]) for m in base]
    print(f"\n[sensitivity] BASE models only (n={len(base)}: "
          f"{', '.join(base)}):")
    print(f"{'task':22s} {'r(above-maj)':>13s} {'r(gap)':>9s} {'p(gap)':>8s}")
    for task in TASKS:
        am = [cells(gate, m, task)["above_maj"] for m in base]
        gp = [cells(gate, m, task)["gap"] for m in base]
        rg = pearson(lp3, gp)
        print(f"{task:22s} {pearson(lp3, am):13.3f} {rg:9.3f} {r_pvalue(rg, len(base)):8.4f}")

    # ---------------- B4: full per-model matrix (compute tasks)
    print("\n" + "=" * 108)
    print("TABLE B4 -- per-model rep / readout / gap, compute tasks, shots=%d" % args.shots)
    print("=" * 108)
    print(f"{'model':24s}{'B':>6s}" + "".join(f"{t[:20]:>30s}" for t in COMPUTE_TASKS))
    print(f"{'':24s}{'':>6s}" + "".join(f"{'rep':>10s}{'out':>10s}{'gap':>10s}"
                                        for _ in COMPUTE_TASKS))
    for m in sorted(models, key=lambda m: (SPEC[m][0], SPEC[m][1])):
        line = f"{m:24s}{SPEC[m][1]:6}"
        for t in COMPUTE_TASKS:
            c = cells(gate, m, t)
            line += f"{c['rep']:10.3f}{c['readout']:10.3f}{c['gap']:10.3f}"
        print(line)

    print("\n[Gemma-2 -- EXCLUDED above, shown for completeness only]")
    for m in sorted(GEMMA):
        line = f"{m:24s}{SPEC[m][1]:6}"
        for t in COMPUTE_TASKS:
            c = cells(gate, m, t)
            line += f"{c['rep']:10.3f}{c['readout']:10.3f}{c['gap']:10.3f}" if c else " " * 30
        print(line)

    if args.latex:
        print("\n" + "=" * 108)
        print("LaTeX (compute tasks, within-family readout gap)")
        print("=" * 108)
        print(r"\begin{tabular}{llrrrrr}")
        print(r"\toprule")
        print(r"Family & Task & \multicolumn{4}{c}{readout gap (rep $-$ readout) by size} & $\Delta$ \\")
        print(r"\midrule")
        for fam in FAMILIES_3PLUS + FAMILIES_2:
            ms = by_fam[fam]
            for i, task in enumerate(COMPUTE_TASKS):
                traj = [cells(gate, m, task)["gap"] for m in ms]
                cellsx = [f"{v:+.2f}" for v in traj] + [""] * (4 - len(traj))
                d = traj[-1] - traj[0]
                fname = fam if i == 0 else ""
                print(f"{fname} & \\texttt{{{task.replace('_', chr(92)+'_')}}} & "
                      + " & ".join(cellsx) + f" & {d:+.2f} \\\\")
            print(r"\midrule")
        print(r"\bottomrule")
        print(r"\end{tabular}")
        print("% sizes per family: " + "; ".join(
            f"{f}: {', '.join(str(SPEC[m][1]) + 'B' for m in by_fam[f])}"
            for f in FAMILIES_3PLUS + FAMILIES_2))

    outp = ROOT / "outputs" / "analysis" / f"within_family_s{args.shots}.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"corr": corr,
                                "per_model": {m: {t: cells(gate, m, t) for t in TASKS}
                                              for m in models}}, indent=2, default=float))
    print(f"\nSaved -> {outp}")


if __name__ == "__main__":
    main()
