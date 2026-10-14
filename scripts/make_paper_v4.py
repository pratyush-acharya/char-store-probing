#!/usr/bin/env python3
"""Figures and LaTeX tables for the paper, generated from the v4 analysis outputs (no GPU).

Every number in a the paper table comes from this script, so the tables cannot drift from the data.
Inputs:  outputs/v4/analysis/{summary,followups,floors_s0,floors_s3}.json, outputs/v4/wordpos/*.json
Outputs: paper/figures/*.pdf (+ .png previews), paper/tables/*.tex,
         paper/tables/numbers.json (every in-text number the generator also computes)

Run:  uv run python scripts/make_paper_v4.py
"""
import glob, json, os, re, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_within_family import SPEC          # noqa: E402
import analyze_v4 as av                         # noqa: E402
import analyze_v4_followups as fu               # noqa: E402

A = ROOT / "outputs/v4/analysis"
FIG, TAB = ROOT / "paper/figures", ROOT / "paper/tables"
S = json.load(open(A / "summary.json"))
F = json.load(open(A / "followups.json"))
FL = {0: json.load(open(A / "floors_s0.json")), 3: json.load(open(A / "floors_s3.json"))}
WP = fu.load("outputs/v4/wordpos")
NUM = {}

# --- style: validated 3-slot categorical palette (light), marker shapes as secondary encoding ---
C = {"q": "#2a78d6", "nq": "#eb6834", "word": "#1baf7a"}
MK = {"q": "o", "nq": "s", "word": "^"}
LBL = {"q": "answer pos., question", "nq": "answer pos., no question", "word": "word tokens, no question"}
INK, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e6e5e1", "#f1f0ec"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.labelcolor": INK,
    "legend.frameon": False, "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.dpi": 220,
    "pdf.fonttype": 42,
})
TT = lambda t: r"\texttt{" + t.replace("_", r"\_") + "}"
f3 = lambda v: "--" if v is None else f"{v:.3f}"
s3 = lambda v: "--" if v is None else f"{v:+.3f}"
pv = lambda p: "--" if p is None else ("$<$0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.001 else f"{p:.3f}")


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def write(name, body):
    # typographic minus for negative numbers (not inside \\cmidrule ranges like {3-6})
    body = re.sub(r"(?<![\w\\{])-(?=\d)", "$-$", body)
    (TAB / f"{name}.tex").write_text(body)


# ============================================================================ figure 1
def fig_scaling():
    """Above-floor accuracy vs scale for the three arms, shots=0, four prompt-sensitive tasks."""
    panels = [("last_letter", "orig", True), ("most_frequent_letter", "cvs", True),
              ("extraction", "orig", False), ("letter_before", "cvs", False),
              ("word_length", "cvs", True)]
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.3), sharey=True)
    for ax, (t, ts, has_word) in zip(axes, panels):
        floor = FL[0][t]["floor"]
        models = sorted({m for (s_, m, h) in WP if s_ == ts and h == 0}, key=lambda m: (SPEC[m][1], m))
        x = np.log10([SPEC[m][1] for m in models])
        arms = [("q", "question", "answer"), ("nq", "neutral", "answer")]
        if has_word:
            read = "word_mean" if t == "most_frequent_letter" else "word_last"
            arms.append(("word", "neutral", read))
        for key, arm, pos in arms:
            y = np.array([WP[(ts, m, 0)]["arms"][arm][pos][t]["acc_heldout_sel"] - floor
                          for m in models])
            ax.scatter(x, y, s=16, marker=MK[key], color=C[key], edgecolor="white", linewidth=0.6,
                       zorder=3, label=LBL[key])
            b, a = np.polyfit(x, y, 1)
            xx = np.array([x.min(), x.max()])
            ax.plot(xx, a + b * xx, color=C[key], lw=1.4, zorder=2)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_title(t, family="monospace", fontsize=7.5)
        ax.set_xticks(np.log10([1, 3, 10]))
        ax.set_xticklabels(["1B", "3B", "10B"])
        ax.set_xlabel("parameters", fontsize=8)
    axes[0].set_ylabel("accuracy above word-only floor")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.10))
    fig.tight_layout()
    save(fig, "fig1_scaling_by_arm")


# ============================================================================ figure 2
def fig_two_phase():
    """Layer profiles, shots=0: answer@question, answer@neutral, word@neutral, small vs large."""
    tasks = [("last_letter", "orig", "word_last"), ("most_frequent_letter", "cvs", "word_mean")]
    models = [("Qwen3_0_6B", "Qwen3-0.6B"), ("phi_4", "phi-4 (14B)")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.3), sharex=True)
    for i, (t, ts, read) in enumerate(tasks):
        floor = FL[0][t]["floor"]
        for j, (m, name) in enumerate(models):
            ax = axes[i, j]
            ax.axvspan(0, 0.25, color=BAND, zorder=0, lw=0)
            ax.axvspan(0.5, 1.0, color=BAND, zorder=0, lw=0)
            d = WP[(ts, m, 0)]
            for key, arm, pos in (("q", "question", "answer"), ("nq", "neutral", "answer"),
                                  ("word", "neutral", read)):
                c = fu.curve(d, arm, pos, t)
                ax.plot([p for p, _ in c], [v for _, v in c], color=C[key], marker=MK[key],
                        ms=4, mec="white", mew=0.5, label=LBL[key], zorder=3)
            ax.axhline(floor, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.text(1.0, floor - 0.035, "word-only floor", color=MUTED, fontsize=7, ha="right",
                    va="top")
            ax.set_ylim(0.0, 1.03)
            ax.set_title(f"{t}  ·  {name}", family="monospace", fontsize=8.5)
            if i == 1:
                ax.set_xlabel("relative layer depth")
            if j == 0:
                ax.set_ylabel("held-out-word probe accuracy")
            if i == 0:
                ax.text(0.125, 0.03, "early", ha="center", color=MUTED, fontsize=7.5, va="bottom")
                ax.text(0.75, 0.03, "late", ha="center", color=MUTED, fontsize=7.5, va="bottom")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, "fig2_two_phase_profiles")


# ============================================================================ tables
def embedding_share(t, ts, sh, read):
    """Share of the word-level above-floor signal already present at layer 0 (mean over models)."""
    floor = FL[sh][t]["floor"]
    ms = [m for (s_, m, h) in WP if s_ == ts and h == sh]
    full = np.mean([WP[(ts, m, sh)]["arms"]["neutral"][read][t]["acc_heldout_sel"] - floor for m in ms])
    l0 = np.mean([max(0.0, WP[(ts, m, sh)]["arms"]["neutral"][read][t]["by_layer"]["0"] - floor)
                  for m in ms])
    return l0 / full if full > 0 else None


def tab_store():
    rows = []
    for t, ts in (("first_letter", "orig"), ("last_letter", "orig"), ("most_frequent_letter", "cvs"),
                  ("word_length", "cvs"), ("count_distinct_lm", "lm")):
        for sh in (0, 3):
            e = S["E1"][f"{t}|s{sh}"]
            c = lambda k: (f"{e[k]['mean_above_floor']:+.3f} ({e[k]['models_above_floor']})"
                           if e.get(k) else "--")
            rows.append(f"{TT(t)} & {sh} & {e['floor']:.3f} & {c('word_nq_last')} & {c('word_nq')} & "
                        f"{c('ans_nq')} & {c('ans_q')} \\\\")
        rows.append(r"\addlinespace[2pt]")
    for t, ts, read in (("last_letter", "orig", "word_last"), ("most_frequent_letter", "cvs", "word_mean"),
                        ("first_letter", "orig", "word_mean")):
        for sh in (0, 3):
            NUM[f"embedding_share|{t}|{read}|s{sh}"] = round(embedding_share(t, ts, sh, read), 3)
    write("tab_store", "\n".join([
        r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r"task & shots & floor & word, last token & word, span mean & answer, no question & answer, question \\",
        r"\midrule", *rows[:-1], r"\bottomrule", r"\end{tabular}"]))


def tab_interaction():
    rows = []
    for sh in (0, 3):
        late_keys = []
        for t, ts in fu.PROMPT_SENSITIVE:
            e5 = S["E5"][f"{t}|s{sh}"]["question_minus_neutral"]
            late = F["F1"][f"s{sh}"][f"{t}|late|q_minus_nq"]
            early = F["F1"][f"s{sh}"][f"{t}|early|q_minus_nq"]
            late_keys.append(e5["p_perm"])
            ci = e5["family_boot_ci95"]
            rows.append([TT(t), str(sh), f"{e5['slope_per_decade']:+.3f}", e5["p_perm"],
                         f"{e5['within_family_slope']:+.3f}", f"[{ci[0]:+.3f}, {ci[1]:+.3f}]",
                         f"{late['slope_per_decade']:+.3f}", pv(late["p_holm4"]),
                         f"{early['slope_per_decade']:+.3f}", pv(early["p_perm"])])
        adj = fu.holm(late_keys)
        for r, a in zip(rows[-4:], adj):
            r[3] = f"{pv(r[3])} ({pv(a)})"
    body = [" & ".join(r) + r" \\" for r in rows]
    body.insert(4, r"\addlinespace[2pt]")
    write("tab_interaction", "\n".join([
        r"\begin{tabular}{@{}lrrrrrrrrr@{}}", r"\toprule",
        r" & & \multicolumn{4}{c}{all layers, held-out-selected} & \multicolumn{2}{c}{late ($\geq$0.5)} & \multicolumn{2}{c}{early ($\leq$0.25)} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}\cmidrule(l){9-10}",
        r"task & shots & slope & $p$ (Holm) & within-fam. & family CI & slope & $p_{\text{Holm}}$ & slope & $p$ \\",
        r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]))


def tab_retained():
    order = ["extraction", "last_letter", "most_frequent_letter", "letter_before", "word_length",
             "counting_lm", "count_distinct_lm", "first_letter", "contains_letter"]
    rows = []
    for t in order:
        cells = [TT(t)]
        for sh in (0, 3):
            e = S["E2"][f"{t}|s{sh}"]
            cells += [f3(e.get("neutral_retained_above_majority")),
                      f3(e.get("neutral_retained_above_floor")),
                      f3(e.get("continuation_retained_above_floor"))]
        rows.append(" & ".join(cells) + r" \\")
    write("tab_retained", "\n".join([
        r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r" & \multicolumn{3}{c}{shots $=0$} & \multicolumn{3}{c}{shots $=3$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"task & majority & neutral & continuation & majority & neutral & continuation \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_nuisance():
    order = ["first_letter", "last_letter", "extraction", "most_frequent_letter", "contains_letter",
             "word_length", "letter_before", "counting", "count_distinct", "counting_lm",
             "count_distinct_lm"]
    names = {"length": "length", "letterset": "letter set", "length+letterset": "length + letter set",
             "letterset+q": "letter set + query", "majority": "majority"}
    rows = []
    for t in order:
        e = S["E6"][f"{t}|s3"]
        rows.append(f"{TT(t)} & {e['majority']:.3f} & {e['floor']:.3f} & {names[e['floor_predictor']]} & "
                    f"{e['models_above_floor']}/{e['n_models']} & {e['best_above_floor']:+.3f} \\\\")
        if t == "letter_before":
            rows.append(r"\addlinespace[2pt]")
    write("tab_nuisance", "\n".join([
        r"\begin{tabular}{@{}lrrlrr@{}}", r"\toprule",
        r"task & majority & floor & floor predictor & models above floor & best above floor \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_crosstask():
    rows = []
    for tag, labels in (("X", ["last_letter", "most_frequent_letter", "word_length", "first_letter"]),
                        ("X_term", ["extraction", "letter_before"])):
        for a in labels:
            e = S[tag][a]
            rows.append(f"{TT(a)} & {e['total']:.3f} & {100*e['frame']:.0f}\\% & {100*e['wrong_q']:.0f}\\% & "
                        f"\\textbf{{{100*e['specific']:.0f}\\%}} & {e['n_cells']} & {e['n_models']} \\\\")
        if tag == "X":
            rows.append(r"\addlinespace[2pt]")
    write("tab_crosstask", "\n".join([
        r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r"probed label & total $\Delta$ & generic frame & wrong-property $Q$ & specific $Q$ & cells & models \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_gap():
    order = ["last_letter", "most_frequent_letter", "word_length", "extraction", "letter_before",
             "counting_lm", "count_distinct_lm"]
    rows = []
    for t in order:
        e = S["E3"][f"{t}|s3"]
        pm = e["per_model"]
        small = np.mean([pm[m]["gap"] for m in ("Qwen3_0_6B", "Llama_3_2_1B", "OLMo_2_0425_1B")])
        big = np.mean([pm[m]["gap"] for m in ("phi_4", "OLMo_2_1124_13B", "gemma_2_9b")])
        f = lambda d: f"{d['r']:+.2f} ({pv(d['p_perm'])})"
        rows.append(f"{TT(t)} & {small:+.3f} & {big:+.3f} & {f(e['gap'])} & {f(e['logit_gap'])} & "
                    f"{f(e['gap_base_only'])} \\\\")
    write("tab_gap", "\n".join([
        r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r"task & gap, 3 smallest & gap, 3 largest & $r$ (raw) & $r$ (logit) & $r$ (base only, raw) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def tab_f3():
    rows = []
    for t in ("last_letter", "most_frequent_letter", "first_letter", "word_length"):
        e = F["F3"][t]
        for pos, name in (("word_last", "last token"), ("word_mean", "span mean")):
            q = e[pos]["qfirst_minus_wfirst"]
            rows.append(f"{TT(t)} & {name} & {q['mean']:+.3f} & {q['n_positive']}/{e['n_models']} & "
                        f"{pv(q['sign_p_two_sided'])} & {e[pos]['wfirst_minus_neutral_mean']:+.3f} & "
                        f"{q['r']:+.2f} ({pv(q['p_perm'])}) \\\\")
    write("tab_f3", "\n".join([
        r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
        r"task & read & \texttt{qfirst}$-$\texttt{wfirst} & positive & sign $p$ & \texttt{wfirst}$-$neutral & $r$ vs scale ($p$) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]))


def numbers():
    """In-text numbers not already in a table."""
    for k in ("s0", "s3"):
        e = S["ACC"][k]
        NUM[f"acc|{k}"] = e
    for t in ("last_letter", "most_frequent_letter"):
        for sh in (0, 3):
            NUM[f"q_minus_store|{t}|s{sh}"] = S["E5"][f"{t}|s{sh}"]["answerq_minus_wordnq_last"]
            NUM[f"q_minus_store_mean|{t}|s{sh}"] = S["E5"][f"{t}|s{sh}"]["answerq_minus_wordnq"]
            NUM[f"late_q_minus_store|{t}|s{sh}"] = F["F1"][f"s{sh}"][f"{t}|late|q_minus_store"]
    NUM["F2"] = {k: {kk: v[kk] for kk in ("n_cross", "r", "p_perm", "median_depth_small_half",
                                           "median_depth_large_half")} for k, v in F["F2"].items()}
    NUM["E4"] = {k: S["E4"][k] for k in S["E4"]}
    NUM["floors"] = {sh: {t: {"majority": v["majority"], "floor": v["floor"], "length": v["length"]}
                          for t, v in FL[sh].items()} for sh in (0, 3)}
    early = {t: F["F1"]["s0"][f"{t}|early|q_level"]["mean"] for t, _ in fu.PROMPT_SENSITIVE}
    NUM["early_q_level_s0"] = early
    (TAB / "numbers.json").write_text(json.dumps(NUM, indent=1))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    fig_scaling()
    fig_two_phase()
    tab_store(); tab_interaction(); tab_retained(); tab_nuisance(); tab_crosstask(); tab_gap(); tab_f3()
    numbers()
    print("wrote", sorted(os.listdir(FIG)), sorted(os.listdir(TAB)))


if __name__ == "__main__":
    main()
