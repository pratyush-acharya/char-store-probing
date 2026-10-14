#!/usr/bin/env python3
"""v4 analysis (LOCAL, CPU only): turns the v4 GPU outputs into the paper's numbers.

Inputs (produced by scripts/run_v4_suite.sh):
  outputs/bigstim_v4_grid/gate_{set}_{model}_s{shots}.json    main grid, fixed estimator (E4)
  outputs/bigstim_v4_grid/align_{set}_{model}_s{shots}.json   accessibility + rank control
  outputs/v4/wordpos/wordpos_{set}_{model}_s{shots}.json      word vs answer position (E1, E2)
  outputs/v4/crosstask/crosstask_{set}_{model}_s0[_term].json cross-task control, fixed estimator
  outputs/bigstim_stage1_gate_grouped{,_cvs}/results_*_s3.json v3 published grid (E4 comparison)

Analyses (each section of summary.md names the experiment it serves):
  floors  word-only nuisance floors for every (task, shots), rebuilt from the v4 sampler
  E1      word@neutral / word@question / answer@neutral / answer@question above the floor,
          scaling correlations, routing ratio
  E2      retained fraction ABOVE THE NUISANCE FLOOR (v3 used above majority)
  E3      ceiling-aware representation-readout gap (logit scale, unsaturated cells, base-only)
  E4      selection optimism (peak - held-out-selected) and v3 -> v4 shifts
  E5      question x scale interaction: slope of (question - no-question) on log10 params with a
          permutation p, a within-family (fixed-effect) slope, and a family-cluster bootstrap CI
  E6      length-matched count tasks against their floors
  X       cross-task additive decomposition under the fixed estimator

Primary estimator = held-out layer selection (`--estimator hsel`); `--estimator peak` reproduces
the v3-style max over layers for comparison.

Run:  uv run python scripts/analyze_v4.py            [--estimator peak] [--perms 10000]
Out:  outputs/v4/analysis/summary.json, outputs/v4/analysis/summary.md
"""
import argparse, glob, json, os, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_stage1 as s1                                   # noqa: E402
import analyze_nuisance_baselines as nb                  # noqa: E402  (fold + predictor helpers)
from analyze_within_family import SPEC, POST_TRAINED     # noqa: E402

OUT = ROOT / "outputs/v4/analysis"
DIRS = {"grid": "outputs/bigstim_v4_grid", "wordpos": "outputs/v4/wordpos",
        "crosstask": "outputs/v4/crosstask"}
SETS = ("orig", "cvs", "lm")
PURE = {"first_letter", "last_letter", "most_frequent_letter", "word_length", "count_distinct",
        "count_distinct_lm"}
LETTER = s1.LETTER_TASKS
ORACLE = {"length": {"word_length"}, "letterset": {"count_distinct"},
          "letterset+q": {"count_distinct", "contains_letter"},
          "length+letterset": {"word_length", "count_distinct"}}


# ----------------------------------------------------------------------------- small stats
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def perm_p(x, y, n=10000, seed=0):
    """Two-sided permutation p for Pearson r (permute x across models)."""
    r = pearson(x, y)
    if r is None:
        return None
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    null = np.array([pearson(rng.permutation(x), y) for _ in range(n)])
    return float((np.sum(np.abs(null) >= abs(r) - 1e-12) + 1) / (n + 1))


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0:
        return None
    return float(np.polyfit(x, y, 1)[0])


def within_family_slope(models, x, y):
    """Fixed-effect slope: demean x and y within family (families with >= 2 models)."""
    fam = np.array([SPEC[m][0] for m in models])
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.array([np.sum(fam == f) >= 2 for f in fam])
    if keep.sum() < 3:
        return None
    xd, yd = x.copy(), y.copy()
    for f in set(fam[keep]):
        m = fam == f
        xd[m] -= x[m].mean(); yd[m] -= y[m].mean()
    xd, yd = xd[keep], yd[keep]
    if xd.std() == 0:
        return None
    return float((xd * yd).sum() / (xd * xd).sum())


def cluster_boot_slope(models, x, y, n=5000, seed=0):
    """95% CI for the OLS slope, resampling FAMILIES with replacement."""
    fam = np.array([SPEC[m][0] for m in models])
    x, y = np.asarray(x, float), np.asarray(y, float)
    fams = sorted(set(fam))
    rng = np.random.RandomState(seed)
    bs = []
    for _ in range(n):
        idx = np.concatenate([np.where(fam == f)[0] for f in rng.choice(fams, len(fams))])
        b = slope(x[idx], y[idx])
        if b is not None:
            bs.append(b)
    if len(bs) < 100:
        return None
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def scaling(models, vals, perms):
    x = np.log10([SPEC[m][1] for m in models])
    return {"n": len(models), "r": rnd(pearson(x, vals)), "p_perm": rnd(perm_p(x, vals, perms)),
            "slope_per_decade": rnd(slope(x, vals))}


def rnd(v, k=4):
    return None if v is None else round(float(v), k)


def logit(p, n=800):
    eps = 1.0 / (2 * n)
    p = min(max(p, eps), 1 - eps)
    return float(np.log(p / (1 - p)))


# ----------------------------------------------------------------------------- floors
def floors(shots):
    """Word-only nuisance floor per task at this shot count, rebuilt from the CURRENT sampler."""
    fp = OUT / f"floors_s{shots}.json"
    if fp.exists():
        return json.load(open(fp))
    words = s1.load_word_pool(nb.WORDS_FILE)
    res = {}
    for ts in SETS:
        ex = s1.build_examples(words, nb.N_PER_TASK, nb.SEED, shots=shots, tokenizer=None,
                               tasks=s1.TASK_SETS[ts])
        by = defaultdict(list)
        for e in ex:
            by[e["task"]].append(e)
        for t in s1.TASK_SETS[ts]:
            base = s1.base_task(t)
            q = base in nb.QUERY_CHAR_TASKS
            rows, y, g, gfold, folds, classes = nb.prepare(by[t])
            r = {"n": len(rows), "n_words": len(set(g)), "n_classes": len(classes)}
            _, r["majority"] = nb.cv_score(rows, y, gfold, folds, nb.make_majority())
            preds = {"length": nb.make_length_only(), "letterset": nb.make_letterset(base),
                     "length+letterset": nb.make_letterset(base, with_query=q, with_length=True)}
            if q:
                preds["letterset+q"] = nb.make_letterset(base, with_query=True)
            for name, fn in preds.items():
                r[name], _ = nb.cv_score(rows, y, gfold, folds, fn)
            ok = {k: v for k, v in r.items() if k in ORACLE and base not in ORACLE[k]}
            best = max(ok, key=ok.get) if ok else None
            r["floor"] = max(r["majority"], ok[best]) if best else r["majority"]
            r["floor_predictor"] = best if best and ok[best] > r["majority"] else "majority"
            res[t] = r
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(fp, "w"), indent=1)
    return res


# ----------------------------------------------------------------------------- loaders
def _load(pattern, rx):
    out = {}
    for fp in glob.glob(str(ROOT / pattern)):
        m = re.match(rx, os.path.basename(fp))
        if m and m.group(2) in SPEC:
            out[(m.group(1), m.group(2), int(m.group(3)))] = json.load(open(fp))
    return out


def load_grid():
    return _load(f"{DIRS['grid']}/gate_*.json", r"gate_(orig|cvs|lm)_(.+)_s(\d+)\.json")


def load_align():
    return _load(f"{DIRS['grid']}/align_*.json", r"align_(orig|cvs|lm)_(.+)_s(\d+)\.json")


def load_wordpos():
    return _load(f"{DIRS['wordpos']}/wordpos_*.json", r"wordpos_(orig|cvs|lm)_(.+)_s(\d+)\.json")


def load_crosstask(tag=""):
    sfx = f"_{tag}" if tag else ""
    return _load(f"{DIRS['crosstask']}/crosstask_*_s0{sfx}.json",
                 rf"crosstask_(orig|cvs)_(.+)_s(\d+){sfx}\.json")


def load_v3(shots=3):
    out = {}
    for sub in ("bigstim_stage1_gate_grouped", "bigstim_stage1_gate_grouped_cvs"):
        for fp in glob.glob(str(ROOT / "outputs" / sub / f"results_*_s{shots}.json")):
            m = re.match(r".*results_(.+)_s\d\.json", fp).group(1)
            for t, g in json.load(open(fp))["gap"].items():
                out[(m, t)] = g
    return out


def arm_acc(r, est):
    """Accuracy of a probe_arm result under the chosen estimator."""
    if not r:
        return None
    if est == "hsel" and r.get("acc_heldout_sel") is not None:
        return r["acc_heldout_sel"]
    return r["acc_peak"]


def gate_acc(g, est):
    if est == "hsel" and g.get("knowledge_answer_heldout_layer_sel") is not None:
        return g["knowledge_answer_heldout_layer_sel"]
    return g["knowledge_answer_peak"]


# ----------------------------------------------------------------------------- analyses
def e1_e2_e5(wp, fl, est, perms):
    """Word vs answer position (E1), retained above floor (E2), interaction tests (E5)."""
    out = {"E1": {}, "E2": {}, "E5": {}}
    for sh in sorted({k[2] for k in wp}):
        F = fl[sh]
        tasks = sorted({t for (ts, m, s), d in wp.items() if s == sh for t in d["summary"]})
        for t in tasks:
            cells = {}
            for (ts, m, s), d in wp.items():
                if s != sh or t not in d["summary"]:
                    continue
                A = d["arms"]
                g = lambda arm, pos: arm_acc((A.get(arm, {}).get(pos) or {}).get(t), est)
                cells[m] = {"ans_q": g("question", "answer"), "ans_nq": g("neutral", "answer"),
                            "ans_cont": g("continuation", "answer"),
                            "word_nq": g("neutral", "word_mean"),
                            "word_nq_last": g("neutral", "word_last"),
                            "word_q": g("question", "word_mean")}
            models = sorted(cells, key=lambda m: (SPEC[m][1], m))
            floor, maj = F[t]["floor"], F[t]["majority"]
            e1 = {"floor": rnd(floor), "floor_predictor": F[t]["floor_predictor"],
                  "majority": rnd(maj), "n_models": len(models)}
            for k in ("ans_q", "ans_nq", "ans_cont", "word_nq", "word_nq_last", "word_q"):
                ms = [m for m in models if cells[m][k] is not None]
                if not ms:
                    continue
                v = np.array([cells[m][k] - floor for m in ms])
                e1[k] = {"mean_above_floor": rnd(v.mean()), "models_above_floor": int((v > 0).sum()),
                         "scaling_above_floor": scaling(ms, v, perms),
                         "per_model": {m: rnd(x) for m, x in zip(ms, v)}}
            # routing ratio: how much of the word-level store reaches the answer position
            rr = {m: (cells[m]["ans_q"] - floor) / (cells[m]["word_nq"] - floor)
                  for m in models if cells[m]["word_nq"] is not None and cells[m]["ans_q"] is not None
                  and cells[m]["word_nq"] - floor > 0.05}
            if rr:
                ms = sorted(rr, key=lambda m: (SPEC[m][1], m))
                e1["routing_ratio"] = {"median": rnd(np.median(list(rr.values()))),
                                       "scaling": scaling(ms, [rr[m] for m in ms], perms),
                                       "per_model": {m: rnd(rr[m]) for m in ms}}
            out["E1"][f"{t}|s{sh}"] = e1

            # E2: retained = mean(nq - base) / mean(q - base), base = floor (v4) or majority (v3)
            e2 = {}
            for carrier, key in (("neutral", "ans_nq"), ("continuation", "ans_cont")):
                ms = [m for m in models if cells[m][key] is not None and cells[m]["ans_q"] is not None]
                if not ms:
                    continue
                q = np.array([cells[m]["ans_q"] for m in ms])
                n = np.array([cells[m][key] for m in ms])
                for base_name, base in (("floor", floor), ("majority", maj)):
                    den = (q - base).mean()
                    e2[f"{carrier}_retained_above_{base_name}"] = rnd((n - base).mean() / den) \
                        if den > 0.02 else None
                e2[f"{carrier}_nq_models_above_floor"] = int(((n - floor) > 0).sum())
                e2[f"{carrier}_q_models_above_floor"] = int(((q - floor) > 0).sum())
            out["E2"][f"{t}|s{sh}"] = e2

            # E5: interaction = slope of (question - no-question) on log10 params
            e5 = {}
            for name, (a, b) in {"question_minus_neutral": ("ans_q", "ans_nq"),
                                 "question_minus_continuation": ("ans_q", "ans_cont"),
                                 "answerq_minus_wordnq": ("ans_q", "word_nq"),
                                 "answerq_minus_wordnq_last": ("ans_q", "word_nq_last")}.items():
                ms = [m for m in models if cells[m][a] is not None and cells[m][b] is not None]
                if len(ms) < 4:
                    continue
                x = np.log10([SPEC[m][1] for m in ms])
                dlt = np.array([cells[m][a] - cells[m][b] for m in ms])
                e5[name] = {"n": len(ms), "slope_per_decade": rnd(slope(x, dlt)),
                            "r": rnd(pearson(x, dlt)), "p_perm": rnd(perm_p(x, dlt, perms)),
                            "within_family_slope": rnd(within_family_slope(ms, x, dlt)),
                            "family_boot_ci95": [rnd(v) for v in
                                                 (cluster_boot_slope(ms, x, dlt) or [None, None])],
                            "base_only": scaling([m for m in ms if m not in POST_TRAINED],
                                                 [cells[m][a] - cells[m][b] for m in ms
                                                  if m not in POST_TRAINED], perms)}
            out["E5"][f"{t}|s{sh}"] = e5
    return out


def e3_e4_e6(grid, fl, v3, est, perms):
    out = {"E3": {}, "E4": {}, "E6": {}}
    for sh in sorted({k[2] for k in grid}):
        F = fl[sh]
        per_task = defaultdict(dict)
        for (ts, m, s), d in grid.items():
            if s == sh:
                for t, g in d["gap"].items():
                    per_task[t][m] = (g, d["meta"])
        for t, cells in sorted(per_task.items()):
            models = sorted(cells, key=lambda m: (SPEC[m][1], m))
            rep = {m: gate_acc(cells[m][0], est) for m in models}
            ro = {m: cells[m][0]["prediction_acc"] for m in models}
            gap = {m: rep[m] - ro[m] for m in models}
            lgap = {m: logit(rep[m]) - logit(ro[m]) for m in models}
            unsat = [m for m in models if rep[m] < 0.95 and ro[m] < 0.95]
            base = [m for m in models if m not in POST_TRAINED]
            out["E3"][f"{t}|s{sh}"] = {
                "gap": scaling(models, [gap[m] for m in models], perms),
                "logit_gap": scaling(models, [lgap[m] for m in models], perms),
                "gap_unsaturated": scaling(unsat, [gap[m] for m in unsat], perms)
                if len(unsat) >= 4 else {"n": len(unsat)},
                "gap_base_only": scaling(base, [gap[m] for m in base], perms),
                "logit_gap_base_only": scaling(base, [lgap[m] for m in base], perms),
                "per_model": {m: {"rep": rnd(rep[m]), "readout": rnd(ro[m]), "gap": rnd(gap[m]),
                                  "logit_gap": rnd(lgap[m])} for m in models}}

            # E4: selection optimism and v3 -> v4 shift
            opt = {m: cells[m][0]["knowledge_answer_peak"]
                   - cells[m][0].get("knowledge_answer_heldout_layer_sel",
                                     cells[m][0]["knowledge_answer_peak"]) for m in models}
            depth = {m: cells[m][1].get("n_layers") for m in models}
            e4 = {"mean_selection_optimism": rnd(np.mean(list(opt.values()))),
                  "optimism_vs_depth_r": rnd(pearson([depth[m] for m in models],
                                                     [opt[m] for m in models])),
                  "optimism_vs_log_params_r": rnd(pearson(np.log10([SPEC[m][1] for m in models]),
                                                          [opt[m] for m in models]))}
            if sh == 3 and v3:
                sh3 = [(m, v3[(m, t)]["knowledge_answer_peak"]) for m in models if (m, t) in v3]
                if sh3:
                    e4["mean_v4_minus_v3_rep"] = rnd(np.mean([rep[m] - v for m, v in sh3]))
            out["E4"][f"{t}|s{sh}"] = e4

            # E6 (and the above-floor verdict for every task)
            floor = F[t]["floor"]
            above = [rep[m] - floor for m in models]
            out["E6"][f"{t}|s{sh}"] = {"floor": rnd(floor), "floor_predictor": F[t]["floor_predictor"],
                                       "majority": rnd(F[t]["majority"]),
                                       "models_above_floor": int(sum(a > 0 for a in above)),
                                       "n_models": len(models),
                                       "best_above_floor": rnd(max(above)),
                                       "scaling_above_floor": scaling(models, above, perms)}
    return out


def accessibility(align, grid):
    """v3 section 8 under v4: answer-token subspace vs rank-matched random and PCA subspaces, and
    the readout-gap-conditioned deficit. One row per shots; cells = (model, task)."""
    out = {}
    for sh in sorted({k[2] for k in align}):
        rows = []
        for (ts, m, s), d in align.items():
            if s != sh:
                continue
            gate = (grid.get((ts, m, s)) or {}).get("gap", {})
            for t, g in d["gap"].items():
                rm = g.get("rank_matched") or {}
                if "proj_minus_rand" not in rm or t not in gate:
                    continue
                rows.append({"proj_minus_rand": rm["proj_minus_rand"],
                             "proj_minus_pca": rm["proj_minus_pca"],
                             "deficit": g["acc_full"] - g["acc_proj"],
                             "acc_full": g["acc_full"],
                             "overlap": g.get("subspace_overlap"),
                             "readout_gap": gate[t]["knowledge_answer_peak"]
                             - gate[t]["prediction_acc"]})
        if not rows:
            continue
        col = lambda k: np.array([r[k] for r in rows], float)
        pr, pp, de, gp = col("proj_minus_rand"), col("proj_minus_pca"), col("deficit"), col("readout_gap")
        terc = np.quantile(gp, [1 / 3, 2 / 3])
        bins = [gp <= terc[0], (gp > terc[0]) & (gp <= terc[1]), gp > terc[1]]
        uns = col("acc_full") < 0.95
        out[f"s{sh}"] = {"n_cells": len(rows),
                         "proj_minus_rand_mean": rnd(pr.mean()), "proj_wins_rand": int((pr > 0).sum()),
                         "proj_minus_pca_mean": rnd(pp.mean()), "proj_wins_pca": int((pp > 0).sum()),
                         "overlap_median": rnd(np.nanmedian(col("overlap"))),
                         "r_gap_deficit": rnd(pearson(gp, de)),
                         "r_gap_deficit_unsaturated": rnd(pearson(gp[uns], de[uns])),
                         "deficit_by_gap_tertile": [rnd(de[b].mean()) for b in bins],
                         "proj_minus_rand_by_gap_tertile": [rnd(pr[b].mean()) for b in bins]}
    return out


def load_cont(tag=""):
    """Same-draw continuation arms (R3): crosstask_{set}_{model}_s0[_term]_cont.json."""
    sfx = f"_{tag}" if tag else ""
    return _load(f"{DIRS['crosstask']}/crosstask_*_s0{sfx}_cont.json",
                 rf"crosstask_(orig|cvs)_(.+)_s(\d+){sfx}_cont\.json")


def crosstask(ct, wp, est, cont=None):
    """Additive decomposition, zero-leak primary cells, shots=0 (fixed estimator). The continuation
    floor comes from the SAME example draw when a _cont file exists (R3); otherwise it is
    joined per (model, set, task) from the wordpos shots=0 files."""
    rows = defaultdict(list)
    cont = cont or {}
    sources = defaultdict(int)
    for (ts, m, sh), d in ct.items():
        w = wp.get((ts, m, 0))
        same = cont.get((ts, m, sh))
        for key, dl in d["delta"].items():
            a, b = dl["probe_label"], dl["asked"]
            if (a in LETTER) == (b in LETTER):          # zero-leak = letter vs digit pairs only
                continue
            self_r, cross_r = d["self"].get(a), d["cross"].get(b, {}).get(a)
            nq_r = d["noquery"].get(a)
            if same is not None:
                cont_r = same["continuation"].get(a)
                src = "same_draw"
            else:
                cont_r = (((w or {}).get("arms", {}).get("continuation", {}).get("answer") or {})
                          .get(a))
                src = "wordpos_join"
            if not (self_r and cross_r and nq_r and cont_r):
                continue
            s_, c_, n_, k_ = (arm_acc(self_r, est), arm_acc(cross_r, est),
                              arm_acc(nq_r, est), arm_acc(cont_r, est))
            if s_ - n_ <= 0.05:                        # usable denominator (v3 rule)
                continue
            rows[a].append({"model": m, "total": s_ - n_, "frame": k_ - n_, "wrong_q": c_ - k_,
                            "specific": s_ - c_})
            sources[src] += 1
    out = {}
    for a, rs in rows.items():
        tot = np.mean([r["total"] for r in rs])
        out[a] = {"n_cells": len(rs), "n_models": len({r["model"] for r in rs}),
                  "continuation_source": dict(sources),
                  "total": rnd(tot),
                  **{k: rnd(np.mean([r[k] for r in rs]) / tot) for k in ("frame", "wrong_q",
                                                                           "specific")}}
    return out


# ----------------------------------------------------------------------------- report
def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    return s + "".join("| " + " | ".join("" if v is None else str(v) for v in r) + " |\n"
                       for r in rows)


def write_md(S, est):
    L = [f"# v4 analysis summary (estimator = {est})\n"]
    L.append("## E1 — word position vs answer position (mean above nuisance floor; models above floor; r vs log10 params)\n")
    rows = []
    for k, e in sorted(S["E1"].items()):
        def c(key):
            v = e.get(key)
            return None if not v else (f"{v['mean_above_floor']:+.3f} ({v['models_above_floor']}/"
                                       f"{e['n_models']}; r={v['scaling_above_floor']['r']})")
        rows.append([k, e["floor"], c("word_nq"), c("word_nq_last"), c("word_q"), c("ans_nq"),
                     c("ans_q"), (e.get("routing_ratio") or {}).get("median")])
    L.append(md_table(["task|shots", "floor", "word@neutral (span mean)",
                       "word@neutral (last token)", "word@question", "answer@neutral",
                       "answer@question", "routing ratio (median, span mean)"], rows))
    L.append("\n## E2 — retained fraction (no-question / question), above floor vs above majority\n")
    rows = [[k, e.get("neutral_retained_above_floor"), e.get("neutral_retained_above_majority"),
             e.get("continuation_retained_above_floor"), e.get("neutral_nq_models_above_floor")]
            for k, e in sorted(S["E2"].items())]
    L.append(md_table(["task|shots", "neutral, above floor", "neutral, above majority",
                       "continuation, above floor", "no-q models above floor"], rows))
    L.append("\n## E5 — question x scale interaction (slope of question − no-question per decade)\n")
    rows = []
    for k, e in sorted(S["E5"].items()):
        v = e.get("question_minus_neutral")
        if v:
            rows.append([k, v["slope_per_decade"], v["r"], v["p_perm"], v["within_family_slope"],
                         v["family_boot_ci95"], (e.get("answerq_minus_wordnq") or {}).get("p_perm"),
                         (e.get("answerq_minus_wordnq_last") or {}).get("p_perm")])
    L.append(md_table(["task|shots", "slope", "r", "p (perm)", "within-family slope",
                       "family-bootstrap 95% CI", "p (ans@q − word@nq, mean)",
                       "p (ans@q − word@nq, last)"], rows))
    L.append("\n## E3 — representation−readout gap vs scale (r, permutation p)\n")
    rows = []
    for k, e in sorted(S["E3"].items()):
        f = lambda d: None if "r" not in d else f"{d['r']} (p={d['p_perm']}, n={d['n']})"
        rows.append([k, f(e["gap"]), f(e["logit_gap"]), f(e["gap_unsaturated"]),
                     f(e["gap_base_only"])])
    L.append(md_table(["task|shots", "gap", "logit gap", "gap, unsaturated", "gap, base only"], rows))
    L.append("\n## E4 — estimator: selection optimism (peak − held-out-selected)\n")
    rows = [[k, e["mean_selection_optimism"], e["optimism_vs_depth_r"],
             e["optimism_vs_log_params_r"], e.get("mean_v4_minus_v3_rep")]
            for k, e in sorted(S["E4"].items())]
    L.append(md_table(["task|shots", "mean optimism", "r vs depth", "r vs log params",
                       "v4 − v3 representation"], rows))
    L.append("\n## E6 — above nuisance floor, every task (grid, question arm)\n")
    rows = [[k, e["majority"], e["floor"], e["floor_predictor"],
             f"{e['models_above_floor']}/{e['n_models']}", e["best_above_floor"]]
            for k, e in sorted(S["E6"].items())]
    L.append(md_table(["task|shots", "majority", "floor", "floor predictor", "models above floor",
                       "best above floor"], rows))
    if S.get("ACC"):
        L.append("\n## ACC — answer-token subspace vs rank-matched comparators (v3 section 8)\n")
        rows = [[k, e["n_cells"], f"{e['proj_minus_rand_mean']:+.3f} ({e['proj_wins_rand']})",
                 f"{e['proj_minus_pca_mean']:+.3f} ({e['proj_wins_pca']})", e["overlap_median"],
                 e["r_gap_deficit"], e["r_gap_deficit_unsaturated"], e["deficit_by_gap_tertile"],
                 e["proj_minus_rand_by_gap_tertile"]] for k, e in sorted(S["ACC"].items())]
        L.append(md_table(["shots", "cells", "proj − random (wins)", "proj − PCA (wins)",
                           "overlap (median)", "r(gap, deficit)", "r, unsaturated",
                           "deficit by gap tertile", "proj − random by gap tertile"], rows))
    for tag in ("X", "X_term"):
        if S.get(tag):
            L.append(f"\n## {tag} — cross-task decomposition (shots=0, zero-leak cells, share of total)\n")
            rows = [[a, e["total"], e["frame"], e["wrong_q"], e["specific"], e["n_cells"],
                     e["n_models"]] for a, e in sorted(S[tag].items())]
            L.append(md_table(["probed label", "total Δ", "generic frame", "wrong-property Q",
                               "specific Q", "cells", "models"], rows))
    L.append("\n## Coverage\n")
    L.append("```\n" + json.dumps(S["coverage"], indent=1) + "\n```\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="hsel", choices=["hsel", "peak"])
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--refresh-floors", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="read the smoke-test outputs (outputs/v4_smoke, bigstim_v4_smoke_grid)")
    args = ap.parse_args()
    global OUT
    if args.smoke:
        DIRS.update(grid="outputs/bigstim_v4_smoke_grid", wordpos="outputs/v4_smoke/wordpos",
                    crosstask="outputs/v4_smoke/crosstask")
        OUT = ROOT / "outputs/v4_smoke/analysis"
    if s1.LEGACY_SET_ORDER:
        raise SystemExit("unset ORTHO_LEGACY_SET_ORDER: v4 outputs use the sorted-set sampler")
    if args.refresh_floors:
        for fp in OUT.glob("floors_s*.json"):
            fp.unlink()

    grid, wp, align = load_grid(), load_wordpos(), load_align()
    ct, ct_term = load_crosstask(), load_crosstask("term")
    shots = sorted({k[2] for k in grid} | {k[2] for k in wp} | {0, 3})
    fl = {sh: floors(sh) for sh in shots}
    S = {"estimator": args.estimator,
         "coverage": {"grid_files": len(grid), "wordpos_files": len(wp), "align_files": len(align),
                      "crosstask_files": len(ct), "crosstask_term_files": len(ct_term),
                      "grid_models": sorted({k[1] for k in grid}),
                      "wordpos_models": sorted({k[1] for k in wp}),
                      "missing_wordpos_models": sorted(set(SPEC) - {k[1] for k in wp}),
                      "missing_grid_models": sorted(set(SPEC) - {k[1] for k in grid})},
         "floors": fl}
    S.update(e1_e2_e5(wp, fl, args.estimator, args.perms) if wp else {"E1": {}, "E2": {}, "E5": {}})
    S.update(e3_e4_e6(grid, fl, load_v3(), args.estimator, args.perms) if grid
             else {"E3": {}, "E4": {}, "E6": {}})
    S["ACC"] = accessibility(align, grid) if align else {}
    S["X"] = crosstask(ct, wp, args.estimator, load_cont()) if ct else {}
    S["X_term"] = crosstask(ct_term, wp, args.estimator, load_cont("term")) if ct_term else {}

    OUT.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.estimator == "hsel" else f"_{args.estimator}"
    json.dump(S, open(OUT / f"summary{sfx}.json", "w"), indent=1)
    (OUT / f"summary{sfx}.md").write_text(write_md(S, args.estimator))
    print(f"wrote {OUT / f'summary{sfx}.json'} and summary{sfx}.md")
    print(json.dumps(S["coverage"], indent=1))


if __name__ == "__main__":
    main()
