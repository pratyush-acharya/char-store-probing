#!/usr/bin/env python3
"""Reviewer follow-ups R1/R2 (LOCAL, CPU). R3 is scored by analyze_v4.py (cross-task section).

R1  For single-token and multi-token query words separately: slope against log10 parameters of
    (answer@question - word-level store), with the store read as in the paper (last token for
    last_letter / first_letter / word_length, span mean for most_frequent_letter; identical for a
    single-token word). Models need >= MIN_N held-out examples in the subset.
R2  MLP vs linear probes at shots=0: mean (MLP - linear) at the word store and at answer@question,
    and the slope of (linear answer@question - MLP store), i.e. does the late excess survive a
    nonlinear reading of the word tokens?

Run:  uv run python scripts/analyze_v4_review.py [--perms 10000]
Out:  outputs/v4/analysis/review.{json,md}
"""
import argparse, glob, json, os, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_v4 as av                                    # noqa: E402
import analyze_v4_followups as fu                          # noqa: E402
from analyze_within_family import SPEC                     # noqa: E402

OUT = ROOT / "outputs/v4/analysis"
STORE_READ = {"last_letter": "word_neutral_last", "first_letter": "word_neutral_last",
              "word_length": "word_neutral_last", "most_frequent_letter": "word_neutral_mean"}
MIN_N = 40


def load(d):
    R = {}
    for fp in glob.glob(str(ROOT / d / "review_*.json")):
        m = re.match(r"review_(orig|cvs)_(.+)_s(\d+)\.json", os.path.basename(fp))
        if m and m.group(2) in SPEC:
            R[(m.group(1), m.group(2), int(m.group(3)))] = json.load(open(fp))
    return R


def r1(R, perms):
    out = {}
    for t, read in STORE_READ.items():
        ts = "orig" if t in ("first_letter", "last_letter") else "cvs"
        for sh in (0, 3):
            for sub in ("single_token", "multi_token", "all"):
                ms, diffs, sizes = [], [], []
                for (s_, m, h), d in R.items():
                    if s_ != ts or h != sh or t not in d["results"]:
                        continue
                    lin = d["results"][t]["linear"]
                    a, w = (lin["answer_question"] or {}).get(sub), (lin[read] or {}).get(sub)
                    if a and w and a["n"] >= MIN_N and w["n"] >= MIN_N:
                        ms.append(m); diffs.append(a["acc"] - w["acc"]); sizes.append(a["n"])
                if len(ms) < 6:
                    continue
                order = sorted(range(len(ms)), key=lambda i: (SPEC[ms[i]][1], ms[i]))
                ms = [ms[i] for i in order]; diffs = [diffs[i] for i in order]
                sizes = [sizes[i] for i in order]
                row = fu.interaction(ms, diffs, perms)
                row.update({"median_n": int(np.median(sizes)), "min_n": int(min(sizes)),
                            "per_model": {m: av.rnd(v) for m, v in zip(ms, diffs)}})
                out[f"{t}|s{sh}|{sub}"] = row
    return out


def r2(R, perms):
    out = {}
    for t, read in STORE_READ.items():
        ts = "orig" if t in ("first_letter", "last_letter") else "cvs"
        ms, rows = [], []
        for (s_, m, h), d in R.items():
            if s_ != ts or h != 0 or "mlp" not in d["results"].get(t, {}):
                continue
            x = d["results"][t]
            rows.append({"lin_store": x["linear"][read]["all"]["acc"],
                         "mlp_store": x["mlp"][read]["acc_heldout_sel"],
                         "lin_ans": x["linear"]["answer_question"]["all"]["acc"],
                         "mlp_ans": x["mlp"]["answer_question"]["acc_heldout_sel"]})
            ms.append(m)
        if len(ms) < 6:
            continue
        order = sorted(range(len(ms)), key=lambda i: (SPEC[ms[i]][1], ms[i]))
        ms, rows = [ms[i] for i in order], [rows[i] for i in order]
        col = lambda k: np.array([r[k] for r in rows])
        out[t] = {"n_models": len(ms),
                  "mlp_minus_linear_store": av.rnd((col("mlp_store") - col("lin_store")).mean()),
                  "mlp_minus_linear_answer": av.rnd((col("mlp_ans") - col("lin_ans")).mean()),
                  "linear_ans_minus_mlp_store": fu.interaction(ms, col("lin_ans") - col("mlp_store"), perms),
                  "mlp_ans_minus_mlp_store": fu.interaction(ms, col("mlp_ans") - col("mlp_store"), perms),
                  "per_model": {m: {k: av.rnd(v) for k, v in r.items()} for m, r in zip(ms, rows)}}
    return out


def md(S):
    L = ["# Reviewer follow-ups R1/R2\n", "## R1 — (answer@question − store) vs scale, by word tokenization\n"]
    L.append(av.md_table(["task|shots|subset", "models", "median n", "mean diff", "slope/decade",
                          "p (perm)", "within-family", "family CI"],
                         [[k, v["n"], v["median_n"], v["mean"], v["slope_per_decade"], v["p_perm"],
                           v["within_family_slope"], v["family_boot_ci95"]] for k, v in S["R1"].items()]))
    L.append("\n## R2 — MLP vs linear probes, shots=0\n")
    L.append(av.md_table(["task", "models", "MLP − linear (store)", "MLP − linear (answer)",
                          "slope (lin ans − MLP store), p", "slope (MLP ans − MLP store), p"],
                         [[t, v["n_models"], v["mlp_minus_linear_store"], v["mlp_minus_linear_answer"],
                           f"{v['linear_ans_minus_mlp_store']['slope_per_decade']} ({v['linear_ans_minus_mlp_store']['p_perm']})",
                           f"{v['mlp_ans_minus_mlp_store']['slope_per_decade']} ({v['mlp_ans_minus_mlp_store']['p_perm']})"]
                          for t, v in S["R2"].items()]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--dir", default="outputs/v4/review")
    ap.add_argument("--out", default=str(OUT / "review"))
    a = ap.parse_args()
    R = load(a.dir)
    S = {"coverage": len(R), "R1": r1(R, a.perms), "R2": r2(R, a.perms)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(S, open(a.out + ".json", "w"), indent=1)
    Path(a.out + ".md").write_text(md(S))
    print(f"wrote {a.out}.md  coverage={len(R)}")


if __name__ == "__main__":
    main()
