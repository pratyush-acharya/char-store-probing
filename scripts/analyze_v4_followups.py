#!/usr/bin/env python3
"""v4 follow-ups F1-F3 (LOCAL, CPU only). Reads existing v4 outputs; no model is loaded.

  F1  distance-confound check for E5: the answer@question curve at shots=0 has an EARLY peak
      (~0.1 relative depth, plausibly the word's spelling copied to the nearby answer token) and a
      LATE peak. The neutral carrier's extraction token sits farther from the word, so an early
      peak could favour the question arm for reasons of distance alone. F1 re-runs the
      question x scale interaction using only LATE layers (relative depth >= 0.5), and separately
      only EARLY layers (<= 0.25).
  F2  crossing depth: the first probed relative depth at which answer@question exceeds the
      model's word-level store (word@neutral, best of last-token / span-mean, held-out-selected),
      against log10 params. Descriptive.
  F3  stem order (pre-registered; paper Appendix B): word@qfirst vs word@wfirst vs
      word@neutral, from outputs/v4/stemorder/ (scripts/run_wordpos.py --stem-order).

Layer-restricted values (F1) use the max over the restricted layers of the fold-mean accuracy; the
v4 grid shows peak-vs-held-out selection optimism <= 0.02, so this is not re-corrected.

Run:  uv run python scripts/analyze_v4_followups.py [--perms 10000]
Out:  outputs/v4/analysis/followups.json, outputs/v4/analysis/followups.md
"""
import argparse, glob, json, os, re, sys
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_v4 as av                                    # noqa: E402
from analyze_within_family import SPEC, POST_TRAINED       # noqa: E402

OUT = ROOT / "outputs/v4/analysis"
PROMPT_SENSITIVE = [("last_letter", "orig"), ("most_frequent_letter", "cvs"),
                    ("extraction", "orig"), ("letter_before", "cvs")]
STORE_TASKS = [("last_letter", "orig"), ("most_frequent_letter", "cvs")]
STEM_TASKS = [("last_letter", "orig"), ("most_frequent_letter", "cvs"),
              ("first_letter", "orig"), ("word_length", "cvs")]


def load(dirname):
    D = {}
    for fp in glob.glob(str(ROOT / dirname / "wordpos_*.json")):
        m = re.match(r"wordpos_(orig|cvs|lm)_(.+)_s(\d+)\.json", os.path.basename(fp))
        if m and m.group(2) in SPEC:
            D[(m.group(1), m.group(2), int(m.group(3)))] = json.load(open(fp))
    return D


def curve(d, arm, pos, t):
    r = d["arms"][arm][pos][t]
    n = d["meta"]["n_hidden"]
    return sorted((int(L) / (n - 1), v) for L, v in r["by_layer"].items())


def restricted(d, arm, pos, t, lo, hi):
    vals = [v for depth, v in curve(d, arm, pos, t) if lo <= depth <= hi]
    return max(vals) if vals else None


def store(d, t):
    w = d["arms"]["neutral"]
    return max(w["word_last"][t]["acc_heldout_sel"], w["word_mean"][t]["acc_heldout_sel"])


def interaction(models, vals, perms):
    x = np.log10([SPEC[m][1] for m in models])
    v = np.asarray(vals, float)
    ci = av.cluster_boot_slope(models, x, v)
    base = [i for i, m in enumerate(models) if m not in POST_TRAINED]
    return {"n": len(models), "mean": av.rnd(v.mean()), "slope_per_decade": av.rnd(av.slope(x, v)),
            "r": av.rnd(av.pearson(x, v)), "p_perm": av.rnd(av.perm_p(x, v, perms)),
            "within_family_slope": av.rnd(av.within_family_slope(models, x, v)),
            "family_boot_ci95": [av.rnd(c) for c in (ci or [None, None])],
            "base_only_r": av.rnd(av.pearson(x[base], v[base])),
            "base_only_p_perm": av.rnd(av.perm_p(x[base], v[base], perms))}


def holm(ps):
    """Holm-adjusted p-values (same order as input)."""
    order = np.argsort(ps)
    adj, run = [None] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def sign_test(diffs):
    """Two-sided exact sign test (zeros dropped)."""
    d = [x for x in diffs if x != 0]
    n, k = len(d), sum(x > 0 for x in d)
    if n == 0:
        return None
    tail = sum(comb(n, i) for i in range(min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def f1(D, perms):
    out = {}
    for sh in (0, 3):
        rows = {}
        for t, ts in PROMPT_SENSITIVE:
            models = sorted({m for (s_, m, h) in D if s_ == ts and h == sh}, key=lambda m: (SPEC[m][1], m))
            for band, (lo, hi) in {"late": (0.5, 1.0), "early": (0.0, 0.25),
                                   "all": (0.0, 1.0)}.items():
                q = [restricted(D[(ts, m, sh)], "question", "answer", t, lo, hi) for m in models]
                n = [restricted(D[(ts, m, sh)], "neutral", "answer", t, lo, hi) for m in models]
                rows[f"{t}|{band}|q_minus_nq"] = interaction(models, np.subtract(q, n), perms)
                if band == "early":
                    rows[f"{t}|early|q_level"] = interaction(models, q, perms)
                if (t, ts) in STORE_TASKS and band in ("late", "all"):
                    s = [store(D[(ts, m, sh)], t) for m in models]
                    rows[f"{t}|{band}|q_minus_store"] = interaction(models, np.subtract(q, s), perms)
        late = [k for k in rows if k.endswith("|late|q_minus_nq")]
        for k, p in zip(late, holm([rows[k]["p_perm"] for k in late])):
            rows[k]["p_holm4"] = av.rnd(p)
        out[f"s{sh}"] = rows
    return out


def f2(D, perms):
    out = {}
    for sh in (0, 3):
        for t, ts in STORE_TASKS:
            models = sorted({m for (s_, m, h) in D if s_ == ts and h == sh}, key=lambda m: (SPEC[m][1], m))
            per = {}
            for m in models:
                d = D[(ts, m, sh)]
                st = store(d, t)
                late = [(depth, v) for depth, v in curve(d, "question", "answer", t) if depth >= 0.25]
                cross = next((depth for depth, v in late if v > st), None)
                per[m] = {"store": av.rnd(st), "crossing_depth": av.rnd(cross)}
            ms = [m for m in models if per[m]["crossing_depth"] is not None]
            x = np.log10([SPEC[m][1] for m in ms])
            y = [per[m]["crossing_depth"] for m in ms]
            rank = lambda a: np.argsort(np.argsort(a)).astype(float)
            out[f"{t}|s{sh}"] = {"n_cross": len(ms), "n_models": len(models),
                                 "r": av.rnd(av.pearson(x, y)), "p_perm": av.rnd(av.perm_p(x, y, perms)),
                                 "spearman": av.rnd(av.pearson(rank(x), rank(y))),
                                 "median_depth_small_half": av.rnd(np.median(y[:len(y) // 2])),
                                 "median_depth_large_half": av.rnd(np.median(y[len(y) // 2:])),
                                 "per_model": per}
    return out


def f3(S, perms):
    if not S:
        return {}
    out = {}
    for t, ts in STEM_TASKS:
        models = sorted({m for (s_, m, h) in S if s_ == ts}, key=lambda m: (SPEC[m][1], m))
        if not models:
            continue
        row = {"n_models": len(models)}
        for pos in ("word_last", "word_mean"):
            g = lambda m, arm: S[(ts, m, 0)]["arms"][arm][pos][t]["acc_heldout_sel"]
            qw = [g(m, "qfirst") - g(m, "wfirst") for m in models]
            wn = [g(m, "wfirst") - g(m, "neutral") for m in models]
            row[pos] = {"qfirst_minus_wfirst": {**interaction(models, qw, perms),
                                                "n_positive": int(sum(v > 0 for v in qw)),
                                                "sign_p_two_sided": av.rnd(sign_test(qw))},
                        "wfirst_minus_neutral_mean": av.rnd(np.mean(wn)),
                        "wfirst_minus_neutral_n_positive": int(sum(v > 0 for v in wn)),
                        "per_model_qfirst_minus_wfirst": {m: av.rnd(v) for m, v in zip(models, qw)}}
        a = lambda m, arm: S[(ts, m, 0)]["arms"][arm]["answer"][t]["acc_heldout_sel"]
        r = lambda m, arm: S[(ts, m, 0)]["arms"][arm]["readout"][t]
        row["answer_probe_qfirst_minus_wfirst"] = av.rnd(np.mean([a(m, "qfirst") - a(m, "wfirst")
                                                                  for m in models]))
        row["readout_qfirst"] = av.rnd(np.mean([r(m, "qfirst") for m in models]))
        row["readout_wfirst"] = av.rnd(np.mean([r(m, "wfirst") for m in models]))
        if t in ("last_letter", "most_frequent_letter"):
            pw = row["word_last"]["qfirst_minus_wfirst"]
            ctrl = row["word_last"]["wfirst_minus_neutral_mean"]
            row["prereg_primary_pass"] = bool(pw["n_positive"] >= 12 and (pw["sign_p_two_sided"] or 1) < 0.05)
            row["prereg_prefix_control_pass"] = bool(abs(ctrl) < 0.02)
        out[t] = row
    return out


def md(R):
    L = ["# v4 follow-ups (F1-F3)\n"]
    for sh, rows in R["F1"].items():
        L.append(f"## F1 — interaction by layer band, {sh}\n")
        L.append(av.md_table(["contrast", "mean", "slope/decade", "r", "p (perm)", "p (Holm, 4)",
                              "within-family", "family CI", "base-only r (p)"],
                             [[k, v["mean"], v["slope_per_decade"], v["r"], v["p_perm"],
                               v.get("p_holm4"), v["within_family_slope"], v["family_boot_ci95"],
                               f"{v['base_only_r']} ({v['base_only_p_perm']})"]
                              for k, v in rows.items()]))
    L.append("\n## F2 — crossing depth vs scale\n")
    L.append(av.md_table(["task|shots", "crossed", "r", "p (perm)", "Spearman",
                          "median depth, smaller 8", "median depth, larger 8"],
                         [[k, f"{v['n_cross']}/{v['n_models']}", v["r"], v["p_perm"], v["spearman"],
                           v["median_depth_small_half"], v["median_depth_large_half"]]
                          for k, v in R["F2"].items()]))
    if R["F3"]:
        L.append("\n## F3 — stem order (pre-registered)\n")
        rows = []
        for t, v in R["F3"].items():
            for pos in ("word_last", "word_mean"):
                q = v[pos]["qfirst_minus_wfirst"]
                rows.append([t, pos, q["mean"], f"{q['n_positive']}/{v['n_models']}",
                             q["sign_p_two_sided"], q["r"], q["p_perm"],
                             v[pos]["wfirst_minus_neutral_mean"],
                             v.get("prereg_primary_pass") if pos == "word_last" else "",
                             v.get("prereg_prefix_control_pass") if pos == "word_last" else ""])
        L.append(av.md_table(["task", "read", "qfirst − wfirst", "positive", "sign p",
                              "r vs scale", "p (perm)", "wfirst − neutral", "primary pass",
                              "prefix control pass"], rows))
        L.append("\n" + av.md_table(["task", "answer probe qfirst − wfirst", "readout qfirst",
                                      "readout wfirst"],
                                     [[t, v["answer_probe_qfirst_minus_wfirst"], v["readout_qfirst"],
                                       v["readout_wfirst"]] for t, v in R["F3"].items()]))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=10000)
    args = ap.parse_args()
    D, S = load("outputs/v4/wordpos"), load("outputs/v4/stemorder")
    R = {"F1": f1(D, args.perms), "F2": f2(D, args.perms), "F3": f3(S, args.perms),
         "coverage": {"wordpos": len(D), "stemorder": len(S)}}
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(OUT / "followups.json", "w"), indent=1)
    (OUT / "followups.md").write_text(md(R))
    print(f"wrote {OUT / 'followups.md'}  coverage={R['coverage']}")


if __name__ == "__main__":
    main()
