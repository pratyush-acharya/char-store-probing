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
    """Layer profiles, shots=0: answer@question, answer@neutral, word@neutral, small vs large.
    One row of four panels (task x model); values from the wordpos runs (fu.curve)."""
    tasks = [("last_letter", "orig", "word_last"), ("most_frequent_letter", "cvs", "word_mean")]
    models = [("Qwen3_0_6B", "Qwen3-0.6B"), ("phi_4", "phi-4 (14B)")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25), sharey=True)
    for i, (t, ts, read) in enumerate(tasks):
        floor = FL[0][t]["floor"]
        for j, (m, name) in enumerate(models):
            ax = axes[2 * i + j]
            ax.axvspan(0, 0.25, color=BAND, zorder=0, lw=0)
            ax.axvspan(0.5, 1.0, color=BAND, zorder=0, lw=0)
            d = WP[(ts, m, 0)]
            for key, arm, pos in (("q", "question", "answer"), ("nq", "neutral", "answer"),
                                  ("word", "neutral", read)):
                c = fu.curve(d, arm, pos, t)
                ax.plot([p for p, _ in c], [v for _, v in c], color=C[key], marker=MK[key],
                        ms=3.4, mec="white", mew=0.4, lw=1.2, label=LBL[key], zorder=3)
            ax.axhline(floor, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
            ax.text(1.0, floor - 0.035, "floor", color=MUTED, fontsize=7, ha="right", va="top")
            ax.set_ylim(0.0, 1.03)
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "", "0.5", "", "1"])
            ax.set_title(f"{t}\n{name}", family="monospace", fontsize=7.5, linespacing=1.15)
            ax.set_xlabel("relative depth", fontsize=8)
            ax.text(0.125, 0.03, "early", ha="center", color=MUTED, fontsize=7, va="bottom")
            ax.text(0.75, 0.03, "late", ha="center", color=MUTED, fontsize=7, va="bottom")
    axes[0].set_ylabel("probe accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig2_two_phase_profiles")


# ============================================================================ shared helpers
# Every figure below reads its values from S (summary.json), F (followups.json), FL (floors) or WP
# (wordpos runs); nothing is typed in by hand. All figures are 7.2 in wide and included at \linewidth,
# so font sizes match the two original figures.
MUTED2, INK2 = "#898781", "#3d3c39"                       # de-emphasis gray, secondary ink
BLUE_RAMP = ["#86b6ef", "#3987e5", "#184f95"]            # ordinal blue ramp (validator: --ordinal)
GREEN_DARK, GREEN_LIGHT = "#13875e", "#5dc196"            # word store: layer 0 / later (validated --ordinal)
MONO = {"family": "monospace", "fontsize": 7.5}
ORDER_MODELS = sorted(SPEC, key=lambda m: (SPEC[m][1], m))
RING = dict(edgecolor="white", linewidth=0.6, zorder=3)


def logp(models):
    return np.log10([SPEC[m][1] for m in models])


def size_axis(ax, label=True):
    ax.set_xticks(np.log10([1, 3, 10]))
    ax.set_xticklabels(["1B", "3B", "10B"])
    if label:
        ax.set_xlabel("parameters", fontsize=8)


def task_rows(ax, tasks, ys, labels=True):
    ax.set_yticks(ys)
    ax.set_yticklabels(tasks if labels else [], **MONO)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)


def ols(ax, x, y, color, lw=1.2):
    b, a = np.polyfit(x, y, 1)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, a + b * xx, color=color, lw=lw, zorder=2)


def minus(s):
    return s.replace("-", "−")


# ============================================================================ figure: overview
def fig_overview():
    """Figure 1. (a) design schematic; (b) per model, last_letter at 0 shots: word-level store
    (S.E1 word_nq_last per_model) and answer position under the question (S.E1 ans_q per_model),
    both above the floor; (c) slope of (question - neutral) per decade of parameters with 95% family
    CI: all layers held-out-selected (S.E5 question_minus_neutral, as Table tab_interaction), late
    and early bands (F.F1 '<task>|late|q_minus_nq', '<task>|early|q_minus_nq')."""
    FW, FH = 7.2, 3.75
    fig = plt.figure(figsize=(FW, FH))
    # ---------------- (a) schematic on a monospace character grid (x unit = one character)
    fa = 0.355
    ax = fig.add_axes([0.0, 1 - fa, 1.0, fa])
    fs = 7.4
    cw = fs * 0.602 / 72                                   # DejaVu Sans Mono advance width (in)
    H = FH * fa
    ax.set_xlim(0, FW / cw)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(0, H - 0.02, "(a)  Design: three prompts per example, same word and label",
            fontsize=8.5, fontweight="bold", va="top", color=INK)
    x0 = 14.5
    rows = [("question", ["The", "last", "letter", "of", "the", "word", '"', "cat", '"', "is"],
             {9: "q"}),
            ("neutral", ["The", "word", '"', "cat", '"', "appears", "in", "this", "sentence", "."],
             {3: "word", 9: "nq"}),
            ("continuation", ["Consider", "the", "word", '"', "cat", '".', "The", "answer", "is"],
             {8: "cont"})]
    ry = [H - 0.33 - r * 0.31 for r in range(3)]
    for (arm, toks, marks), y in zip(rows, ry):
        ax.text(x0 - 1.4, y, arm, ha="right", va="center", fontsize=8, color=INK)
        x = x0
        for i, tok in enumerate(toks):
            ax.text(x + len(tok) / 2, y, tok, ha="center", va="center", family="monospace",
                    fontsize=fs, color=INK,
                    bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.15",
                              fc="#d6efe3" if tok == "cat" else BAND, ec="none"))
            if i in marks:
                xm, ym = x + len(tok) / 2, y - 0.145
                if marks[i] == "cont":
                    ax.plot(xm, ym, marker="s", ms=4.6, mfc="white", mec=C["nq"], mew=1.0)
                else:
                    k = marks[i]
                    ax.plot(xm, ym, marker=MK[k], ms=5.4, color=C[k], mec="white", mew=0.5)
            x += len(tok) + 1.2
    # key (right of the prompts); it doubles as the legend for panel (b)
    kx = 66
    keys = [("word", "word position: the query word's tokens, no question"),
            ("q", "answer position (last token) under the question"),
            ("nq", "answer position without a question")]
    for (k, txt), y in zip(keys, ry):
        ax.plot(kx, y, marker=MK[k], ms=5.4, color=C[k], mec="white", mew=0.5)
        if k == "nq":
            ax.plot(kx + 2.2, y, marker="s", ms=4.6, mfc="white", mec=C["nq"], mew=1.0)
        ax.text(kx + (4.2 if k == "nq" else 2.0), y, txt, va="center", fontsize=7.6, color=INK)
    ax.text(kx - 1.0, ry[2] - 0.27,
            "Linear probe at 12 depths, held-out words;\naccuracy reported above a word-only floor.",
            va="center", fontsize=7.2, color=MUTED, linespacing=1.25)

    # ---------------- (b) per-model store vs answer position, last_letter, 0 shots
    bot, hgt = 0.11, 0.37
    axb = fig.add_axes([0.075, bot, 0.30, hgt])
    e = S["E1"]["last_letter|s0"]
    ms = ORDER_MODELS
    x = logp(ms)
    st = np.array([e["word_nq_last"]["per_model"][m] for m in ms])
    aq = np.array([e["ans_q"]["per_model"][m] for m in ms])
    axb.vlines(x, np.minimum(st, aq), np.maximum(st, aq), color="#d5d3cc", lw=1.4, zorder=1)
    for key, y in (("word", st), ("q", aq)):
        ols(axb, x, y, C[key])
        axb.scatter(x, y, s=20, marker=MK[key], color=C[key], **RING)
    i0, i1 = ms.index("Qwen3_0_6B"), ms.index("phi_4")
    axb.annotate("Qwen3-0.6B", (x[i0], min(st[i0], aq[i0]) - 0.012), (x[i0] + 0.02, 0.17),
                 fontsize=6.8, color=MUTED, ha="left", va="bottom",
                 arrowprops=dict(arrowstyle="-", color=MUTED2, lw=0.6))
    axb.annotate("phi-4", (x[i1], aq[i1] + 0.012), (x[i1] - 0.05, 0.505), fontsize=6.8,
                 color=MUTED, ha="right", va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED2, lw=0.6))
    axb.text(0.62, 0.46, "answer position,\nquestion", fontsize=7.2, color=INK, ha="right",
             va="center", linespacing=1.1)
    bs, as_ = np.polyfit(x, st, 1)
    axb.annotate("word-level store", (0.36, as_ + bs * 0.36), (0.36, 0.205), fontsize=7.2,
                 color=INK, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED2, lw=0.6, shrinkA=1, shrinkB=2))
    axb.set_ylim(0.15, 0.53)
    axb.set_xlim(x.min() - 0.08, x.max() + 0.08)
    size_axis(axb)
    axb.set_ylabel("accuracy above floor", fontsize=8)
    fig.text(0.0, bot + hgt + 0.09, "(b)  last_letter, 0 shots, one pair per model",
             fontsize=8.5, fontweight="bold", color=INK)

    # ---------------- (c) question x scale slope by depth band, both shot counts
    tasks = [t for t, _ in fu.PROMPT_SENSITIVE]
    bands = [("all layers", "all", "D", INK2, INK2, 0.25),
             ("late (≥ 0.5 depth)", "late", "o", C["q"], C["q"], 0.0),
             ("early (≤ 0.25 depth)", "early", "o", MUTED2, "white", -0.25)]
    axs = [fig.add_axes([0.618, bot, 0.18, hgt]), fig.add_axes([0.812, bot, 0.18, hgt])]
    ys = np.arange(len(tasks))[::-1].astype(float)
    for ax_, sh in zip(axs, (0, 3)):
        ax_.axvline(0, color=MUTED, lw=0.8, zorder=1)
        for label, band, mk, ec, fc, dy in bands:
            for t, y in zip(tasks, ys):
                d = (S["E5"][f"{t}|s{sh}"]["question_minus_neutral"] if band == "all"
                     else F["F1"][f"s{sh}"][f"{t}|{band}|q_minus_nq"])
                lo, hi = d["family_boot_ci95"]
                ax_.plot([lo, hi], [y + dy] * 2, color=ec, lw=1.1, zorder=2,
                         solid_capstyle="round")
                hollow = fc == "white"
                ax_.plot(d["slope_per_decade"], y + dy, marker=mk, ms=4.6 if mk == "o" else 4.0,
                         mfc=fc, mec=ec if hollow else "white", mew=0.9 if hollow else 0.5,
                         zorder=3, ls="none", label=label if t == tasks[0] else None)
        ax_.set_xlim(-0.12, 0.32)
        ax_.set_xticks([0, 0.1, 0.2, 0.3])
        ax_.set_xticklabels(["0", "0.1", "0.2", "0.3"])
        ax_.set_ylim(-0.55, len(tasks) - 0.45)
        ax_.set_title(f"{sh} shots", fontsize=8, color=INK, pad=3)
        ax_.grid(axis="y", visible=False)
    task_rows(axs[0], tasks, ys)
    task_rows(axs[1], tasks, ys, labels=False)
    axs[0].set_xlabel(" ", fontsize=8)
    fig.text(0.805, 0.012, "slope per decade of parameters (question − no question)",
             ha="center", fontsize=8, color=INK)
    fig.text(0.415, bot + hgt + 0.09, "(c)  growth of the question effect with scale, by depth",
             fontsize=8.5, fontweight="bold", color=INK)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=3, bbox_to_anchor=(0.41, bot + hgt + 0.03),
               fontsize=7.2, handletextpad=0.2, columnspacing=0.9)
    save(fig, "fig_overview")


# ============================================================================ figure: store
STORE_TASKS = ["first_letter", "last_letter", "most_frequent_letter", "word_length",
               "count_distinct_lm"]
EMB = [("last_letter", "orig", "word_last", "last token"),
       ("most_frequent_letter", "cvs", "word_mean", "span mean"),
       ("first_letter", "orig", "word_mean", "span mean")]


def fig_store():
    """(a) S.E1 mean_above_floor for word (last token, span mean) and answer (no question,
    question) positions, as Table tab_store; (b) layer-0 part of the word-level store
    (embedding_parts; the printed share equals numbers.json embedding_share)."""
    fig = plt.figure(figsize=(7.2, 2.2))
    cells = [("word_nq_last", "word tokens, last token", "^", C["word"], False),
             ("word_nq", "word tokens, span mean", "^", C["word"], True),
             ("ans_nq", LBL["nq"], "s", C["nq"], False),
             ("ans_q", LBL["q"], "o", C["q"], False)]
    offs = [0.27, 0.09, -0.09, -0.27]
    ys = np.arange(len(STORE_TASKS))[::-1].astype(float)
    bot, hgt = 0.15, 0.58
    axs = [fig.add_axes([0.19, bot, 0.205, hgt]), fig.add_axes([0.41, bot, 0.205, hgt])]
    for ax, sh in zip(axs, (0, 3)):
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        for (k, lab, mk, col, hollow), dy in zip(cells, offs):
            v = [S["E1"][f"{t}|s{sh}"][k]["mean_above_floor"] for t in STORE_TASKS]
            ax.plot(v, ys + dy, marker=mk, ms=5.4 if mk == "^" else 4.8, mfc="white" if hollow else col,
                    mec=col if hollow else "white", mew=1.0 if hollow else 0.5, ls="none",
                    zorder=3, label=lab)
        ax.set_xlim(-0.1, 0.6)
        ax.set_xticks([0, 0.2, 0.4])
        ax.set_ylim(-0.55, len(ys) - 0.45)
        ax.set_title(f"{sh} shots", fontsize=8, color=INK, pad=3)
        ax.set_xlabel("accuracy above floor", fontsize=8)
    task_rows(axs[0], STORE_TASKS, ys)
    task_rows(axs[1], STORE_TASKS, ys, labels=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=2, bbox_to_anchor=(0.07, 0.775), fontsize=7.2,
               handletextpad=0.2, columnspacing=1.2, labelspacing=0.3)
    fig.text(0.0, 0.93, "(a)  mean over the 16 models, by position and prompt", fontsize=8.5,
             fontweight="bold", color=INK, va="bottom")
    # (b) layer-0 share of the word-level store
    ax = fig.add_axes([0.715, bot, 0.27, hgt])
    y, yt, ylab = 0.0, [], []
    for t, ts, read, rname in EMB:
        ax.text(0.0, y + 0.4, t, fontsize=6.8, family="monospace", color=INK, va="bottom")
        ax.text(0.5, y + 0.4, rname, fontsize=6.8, color=MUTED, va="bottom", ha="right")
        for sh in (0, 3):
            l0, full = embedding_parts(t, ts, sh, read)
            ax.barh(y, l0, height=0.6, color=GREEN_DARK, edgecolor="white", linewidth=1.0)
            ax.barh(y, full - l0, left=l0, height=0.6, color=GREEN_LIGHT, edgecolor="white",
                    linewidth=1.0)
            ax.text(full + 0.012, y, f"{100 * l0 / full:.0f}%", va="center", fontsize=6.8,
                    color=INK)
            yt.append(y)
            ylab.append(f"{sh} shots")
            y -= 0.72
        y -= 0.62
    ax.set_yticks(yt)
    ax.set_yticklabels(ylab, fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(y + 0.75, 0.95)
    ax.set_xlim(0, 0.5)
    ax.set_xticks([0, 0.2, 0.4])
    ax.set_xlabel("word-level store above floor", fontsize=8)
    fig.text(0.64, 0.93, "(b)  share already at layer 0", fontsize=8.5, fontweight="bold",
             color=INK, va="bottom")
    from matplotlib.patches import Patch
    fig.legend([Patch(color=GREEN_DARK), Patch(color=GREEN_LIGHT)],
               ["layer 0 (embedding)", "later layers"], loc="lower left", ncol=2,
               bbox_to_anchor=(0.705, 0.775), fontsize=7.2, handlelength=1.0, handletextpad=0.3,
               columnspacing=0.8)
    save(fig, "fig_store")


# ============================================================================ figure: retained
RET_TASKS = ["extraction", "last_letter", "most_frequent_letter", "letter_before", "word_length",
             "counting_lm", "count_distinct_lm", "first_letter", "contains_letter"]


def fig_retained():
    """Retained fraction, as Table tab_retained (S.E2): neutral carrier against the majority
    baseline and against the floor; continuation carrier against the floor."""
    fig = plt.figure(figsize=(7.2, 2.0))
    ys = np.arange(len(RET_TASKS))[::-1].astype(float)
    ys[5:] -= 0.6                                         # gap after the question-driven group
    bot, hgt = 0.16, 0.68
    axs = [fig.add_axes([0.205, bot, 0.385, hgt]), fig.add_axes([0.607, bot, 0.385, hgt])]
    for ax, sh in zip(axs, (0, 3)):
        for xv in (0, 1):
            ax.axvline(xv, color=MUTED, lw=0.8, zorder=1)
        for t, y in zip(RET_TASKS, ys):
            e = S["E2"][f"{t}|s{sh}"]
            maj, fl = e.get("neutral_retained_above_majority"), e.get("neutral_retained_above_floor")
            co = e.get("continuation_retained_above_floor")
            first = t == RET_TASKS[0]
            if maj is not None and fl is not None:
                ax.annotate("", (fl, y), (maj, y), zorder=2,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                            color="#bdbab2", lw=1.2, shrinkA=3, shrinkB=3))
            if maj is not None:
                ax.plot(maj, y, marker="s", ms=4.8, mfc="white", mec=C["nq"], mew=1.0, zorder=3,
                        ls="none", label="neutral, against majority" if first else None)
            if fl is not None:
                ax.plot(fl, y, marker="s", ms=4.8, mfc=C["nq"], mec="white", mew=0.5, zorder=4,
                        ls="none", label="neutral, against floor" if first else None)
            if co is not None:
                ax.plot(co, y, marker="D", ms=3.8, mfc=MUTED2, mec="white", mew=0.5, zorder=3,
                        ls="none", label="continuation, against floor" if first else None)
        ax.set_xlim(-0.1, 1.27)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_ylim(ys.min() - 0.6, ys.max() + 0.6)
        ax.set_title(f"{sh} shots", fontsize=8, color=INK, pad=3)
        ax.set_xlabel("retained fraction of the question arm's signal", fontsize=8)
    task_rows(axs[0], RET_TASKS, ys)
    task_rows(axs[1], RET_TASKS, ys, labels=False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=3, bbox_to_anchor=(0.2, 0.89), fontsize=7.2,
               handletextpad=0.25, columnspacing=1.2)
    save(fig, "fig_retained")


# ============================================================================ figure: cross-task
def fig_crosstask():
    """Decomposition of the question effect, as Table tab_crosstask (S.X, S.X_term): generic frame,
    wrong-property question and specific question as shares of the total Delta."""
    fig = plt.figure(figsize=(7.2, 1.95))
    bot, hgt = 0.21, 0.6
    ax = fig.add_axes([0.205, bot, 0.62, hgt])
    rows = [("X", a) for a in ("last_letter", "most_frequent_letter", "word_length", "first_letter")] \
        + [("X_term", a) for a in ("extraction", "letter_before")]
    ys = np.arange(len(rows))[::-1].astype(float)
    ys[4:] -= 0.6
    parts = [("frame", "generic frame"), ("wrong_q", "wrong-property question"),
             ("specific", "specific question")]
    for (tag, a), y in zip(rows, ys):
        e = S[tag][a]
        pos = 0.0
        for (k, lab), col in zip(parts, BLUE_RAMP):
            v = 100 * e[k]
            if v < 0:                                    # a negative share sits left of zero
                ax.barh(y, v, left=0, height=0.6, color=col, edgecolor="white", linewidth=1.2)
                ax.text(v - 1.2, y, minus(f"{v:.0f}%"), ha="right", va="center", fontsize=6.8,
                        color=INK)
                continue
            ax.barh(y, v, left=pos, height=0.6, color=col, edgecolor="white", linewidth=1.2,
                    label=lab if a == "last_letter" else None)
            if v >= 7:
                ax.text(pos + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=6.8,
                        color=INK if col == BLUE_RAMP[0] else "white",
                        fontweight="bold" if k == "specific" else "normal")
            pos += v
        ax.text(112, y, f"{e['total']:.3f}", va="center", ha="left", fontsize=7, color=INK)
        ax.text(125, y, f"{e['n_models']}", va="center", ha="left", fontsize=7, color=INK)
    ax.text(112, ys[0] + 0.5, "total Δ", fontsize=7, color=MUTED, va="bottom")
    ax.text(125, ys[0] + 0.5, "models", fontsize=7, color=MUTED, va="bottom")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlim(-12, 107)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(ys.min() - 0.5, ys.max() + 0.5)
    ax.set_xlabel("share of the total question effect (self − neutral), zero shots", fontsize=8)
    task_rows(ax, [a for _, a in rows], ys)
    ax.spines["left"].set_visible(False)
    ax.text(-1.5, (ys[3] + ys[4]) / 2, "secondary arm", fontsize=6.6, ha="right",
            color=MUTED, style="italic", va="center")
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=3, bbox_to_anchor=(0.2, 0.85), fontsize=7.2,
               handlelength=1.2, handletextpad=0.35, columnspacing=1.2)
    save(fig, "fig_crosstask")


# ============================================================================ figure: nuisance
NUIS_TASKS = ["first_letter", "last_letter", "extraction", "most_frequent_letter", "contains_letter",
              "word_length", "letter_before", "counting", "count_distinct", "counting_lm",
              "count_distinct_lm"]


def fig_nuisance():
    """Three shots. Per model: question-arm answer-position accuracy above the floor (S.E1
    '<task>|s3' ans_q per_model). Per task: majority baseline relative to the floor (S.E6 majority
    minus S.E6 floor, a difference of two Table tab_nuisance columns) and models above the floor."""
    fig = plt.figure(figsize=(7.2, 2.25))
    bot, hgt = 0.155, 0.73
    ax = fig.add_axes([0.205, bot, 0.70, hgt])
    ys = np.arange(len(NUIS_TASKS))[::-1].astype(float)
    ys[7:] -= 0.6
    for t, y in zip(NUIS_TASKS, ys):
        e6 = S["E6"][f"{t}|s3"]
        pm = S["E1"][f"{t}|s3"]["ans_q"]["per_model"]
        v = np.array([pm[m] for m in ORDER_MODELS])
        first = t == NUIS_TASKS[0]
        ax.scatter(v, np.full(len(v), y), s=16, marker="o", color=C["q"],
                   label="one model: answer position under the question" if first else None, **RING)
        mj = e6["majority"] - e6["floor"]
        if mj < -1e-9:
            ax.plot(mj, y, marker="|", ms=8, mew=1.5, color=INK2, ls="none", zorder=4,
                    label="majority baseline" if first else None)
        ax.text(0.785, y, f"{e6['models_above_floor']}/{e6['n_models']}", va="center",
                fontsize=7.2, color=INK)
    ax.text(0.785, ys[0] + 0.7, "above floor", fontsize=7, color=MUTED, va="bottom")
    ax.axvline(0, color=INK2, lw=1.0, zorder=1)
    ax.text(0.01, ys.min() - 1.0, "word-only floor", fontsize=7, color=INK2, va="center")
    ax.text(-0.49, (ys[6] + ys[7]) / 2, "count tasks",
            fontsize=6.6, color=MUTED, style="italic", va="center")
    ax.set_xlim(-0.5, 0.76)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["−0.4", "−0.2", "0", "0.2", "0.4", "0.6"])
    ax.set_ylim(ys.min() - 1.35, ys.max() + 0.6)
    ax.set_xlabel("accuracy relative to the word-only floor, three shots", fontsize=8)
    task_rows(ax, NUIS_TASKS, ys)
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", ncol=2, bbox_to_anchor=(0.2, 0.9), fontsize=7.2,
               handletextpad=0.3, columnspacing=1.5)
    save(fig, "fig_nuisance")


# ============================================================================ figure: readout gap
GAP_TASKS = ["last_letter", "most_frequent_letter", "word_length", "extraction", "letter_before",
             "counting_lm", "count_distinct_lm"]


def fig_gap():
    """Three shots, per model (S.E3 '<task>|s3' per_model rep / readout / gap). Panel 1: last_letter
    probe and readout accuracy (no fit line). Other panels: raw gap with an OLS line; r on the raw and logit
    scales from S.E3 gap.r and logit_gap.r (uncorrected; p-values in Table tab_gap)."""
    fig = plt.figure(figsize=(7.2, 2.35))
    ms = ORDER_MODELS
    x = logp(ms)
    L, R, B, T, gx, gy = 0.075, 0.995, 0.13, 0.93, 0.035, 0.13
    w, h = (R - L - 3 * gx) / 4, (T - B - gy) / 2
    axes = [[fig.add_axes([L + j * (w + gx), B + (1 - i) * (h + gy), w, h]) for j in range(4)]
            for i in range(2)]
    flat = [a for row in axes for a in row]
    ax = flat[0]
    pm = S["E3"]["last_letter|s3"]["per_model"]
    for key, lab, mk, col in (("rep", "probe", "o", C["q"]), ("readout", "readout", "D", INK2)):
        y = np.array([pm[m][key] for m in ms])
        ax.scatter(x, y, s=15, marker=mk, color=col, label=lab, **RING)
    ax.set_ylim(0, 1.05)
    ax.set_title("last_letter: accuracy", family="monospace", fontsize=7.5, pad=3)
    ax.legend(loc="lower right", fontsize=7, handletextpad=0.1, borderaxespad=0.2)
    ax.set_ylabel("accuracy", fontsize=8)
    for ax, t in zip(flat[1:], GAP_TASKS):
        e = S["E3"][f"{t}|s3"]
        y = np.array([e["per_model"][m]["gap"] for m in ms])
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ols(ax, x, y, C["q"], lw=1.1)
        ax.scatter(x, y, s=15, marker="o", color=C["q"], **RING)
        ax.set_title(f"{t}: gap", family="monospace", fontsize=7.5, pad=3)
        ax.set_ylim(-0.12, 0.66)
        ax.set_yticks([0, 0.2, 0.4, 0.6])
        r_raw, r_log = (minus(f"{e[k]['r']:+.2f}") for k in ("gap", "logit_gap"))
        ax.text(0.97, 0.96, f"r: {r_raw} raw, {r_log} logit", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.6, color=MUTED)
        if ax not in (flat[1], flat[4]):
            ax.tick_params(labelleft=False)
    flat[4].set_ylabel("probe − readout", fontsize=8)
    for ax in flat:
        ax.set_xlim(x.min() - 0.1, x.max() + 0.1)
        size_axis(ax, label=ax in axes[1])
        if ax in axes[0]:
            ax.tick_params(labelbottom=False)
    save(fig, "fig_gap")

# ============================================================================ tables
def embedding_parts(t, ts, sh, read):
    """(layer-0 part, full store): means over models of the word-level above-floor accuracy at
    layer 0 (clipped at 0) and at the held-out-selected layer."""
    floor = FL[sh][t]["floor"]
    ms = [m for (s_, m, h) in WP if s_ == ts and h == sh]
    full = np.mean([WP[(ts, m, sh)]["arms"]["neutral"][read][t]["acc_heldout_sel"] - floor for m in ms])
    l0 = np.mean([max(0.0, WP[(ts, m, sh)]["arms"]["neutral"][read][t]["by_layer"]["0"] - floor)
                  for m in ms])
    return l0, full


def embedding_share(t, ts, sh, read):
    """Share of the word-level above-floor signal already present at layer 0 (mean over models)."""
    l0, full = embedding_parts(t, ts, sh, read)
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
    fig_overview(); fig_store(); fig_retained(); fig_crosstask(); fig_nuisance(); fig_gap()
    tab_store(); tab_interaction(); tab_retained(); tab_nuisance(); tab_crosstask(); tab_gap(); tab_f3()
    numbers()
    print("wrote", sorted(os.listdir(FIG)), sorted(os.listdir(TAB)))


if __name__ == "__main__":
    main()
