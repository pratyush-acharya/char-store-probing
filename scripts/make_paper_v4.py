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
    """Figure 'twophase'. (a) Layer profiles, shots=0: answer@question, answer@neutral,
    word@neutral, smallest vs largest model; values from the wordpos runs (fu.curve).
    (b) growth of the question effect with scale by depth band (draw_slopes)."""
    tasks = [("last_letter", "orig", "word_last"), ("most_frequent_letter", "cvs", "word_mean")]
    models = [("Qwen3_0_6B", "Qwen3-0.6B"), ("phi_4", "phi-4 (14B)")]
    FH = 3.1
    fig = plt.figure(figsize=(7.2, FH))
    top_b, top_h = 0.585, 0.24                  # profile row
    L, R, g = 0.075, 0.995, 0.012
    w = (R - L - 3 * g) / 4
    axes = [fig.add_axes([L + j * (w + g), top_b, w, top_h]) for j in range(4)]
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
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_yticklabels(["0", "0.5", "1"])
            ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xticklabels(["0", "", "0.5", "", "1"])
            ax.set_title(f"{t}\n{name}", family="monospace", fontsize=7.5, linespacing=1.15, pad=3)
            ax.set_xlabel("relative depth", fontsize=8, labelpad=1)
            ax.text(0.125, 0.03, "early", ha="center", color=MUTED, fontsize=7, va="bottom")
            ax.text(0.75, 0.03, "late", ha="center", color=MUTED, fontsize=7, va="bottom")
            if j + i > 0:
                ax.tick_params(labelleft=False)
    axes[0].set_ylabel("probe accuracy")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", ncol=3, bbox_to_anchor=(0.27, 0.965),
               handletextpad=0.4, columnspacing=1.0, fontsize=8)
    fig.text(0.0, 0.965, "(a)  layer profiles, zero shots", fontsize=8.5, fontweight="bold",
             color=INK, va="center")
    # (b) slopes by depth band, all 16 models
    fig.text(0.0, 0.43, "(b)  growth of the question effect with scale, by depth (16 models)",
             fontsize=8.5, fontweight="bold", color=INK, va="center")
    draw_slopes(fig, 0.205, 0.995, 0.14, 0.345, FH)
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
PROMPTS = [
    ("question", ["The", "last", "letter", "of", "the", "word", '"', "cat", '"', "is"], {9: "q"}),
    ("neutral", ["The", "word", '"', "cat", '"', "appears", "in", "this", "sentence", "."],
     {3: "word", 9: "nq"}),
    ("continuation", ["Consider", "the", "word", '"', "cat", '".', "The", "answer", "is"],
     {8: "cont"})]


def prompt_key(ax, x0, ys, fs, gap, rows=PROMPTS):
    """Draw the three prompts on a monospace character grid (x unit = one character) and return
    the probed positions as (x, y, kind). A design schematic: no data."""
    out = []
    for (arm, toks, marks), y in zip(rows, ys):
        x = x0
        for i, tok in enumerate(toks):
            ax.text(x + len(tok) / 2, y, tok, ha="center", va="center", family="monospace",
                    fontsize=fs, color=INK,
                    bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.15",
                              fc="#d6efe3" if tok == "cat" else BAND, ec="none"))
            if i in marks:
                out.append((x + len(tok) / 2, y, marks[i]))
            x += len(tok) + gap
    return out


def site_marker(ax, x, y, k, ms=5.6, color=None):
    if k == "cont":
        ax.plot(x, y, marker="s", ms=ms - 0.8, mfc="white", mec=C["nq"], mew=1.0, ls="none",
                clip_on=False)
    else:
        ax.plot(x, y, marker=MK[k], ms=ms, color=color or C[k], mec="white", mew=0.5, ls="none",
                clip_on=False)


def fig_overview():
    """Figure 1, one chart: last_letter at 0 shots, one column of points per model against size.
    Word-level store: S.E1 'last_letter|s0' word_nq_last per_model (held-out layer selection).
    Answer position, question and no question, restricted to late (>= 0.5) or early (<= 0.25)
    relative depth: fu.restricted on the wordpos runs, the same per-model values whose slopes F1
    reports (F.F1 's0' 'last_letter|late|q_minus_nq', '|early|q_level'). All minus the floor
    (FL[0]). The prompt key on the right is a design schematic and carries no data."""
    t, ts = "last_letter", "orig"
    fl = FL[0][t]["floor"]
    ms = ORDER_MODELS
    x = logp(ms)
    store = np.array([S["E1"][f"{t}|s0"]["word_nq_last"]["per_model"][m] for m in ms])
    band = lambda arm, lo, hi: np.array([fu.restricted(WP[(ts, m, 0)], arm, "answer", t, lo, hi) - fl
                                         for m in ms])
    q_late, q_early = band("question", 0.5, 1.0), band("question", 0.0, 0.25)
    nq_late = band("neutral", 0.5, 1.0)
    # sanity: these are the per-model values behind the F1 interaction statistics
    assert abs(np.mean(q_early + fl) - F["F1"]["s0"][f"{t}|early|q_level"]["mean"]) < 1e-3
    assert abs(np.mean(q_late - nq_late) - F["F1"]["s0"][f"{t}|late|q_minus_nq"]["mean"]) < 1e-3

    FW, FH = 7.2, 2.1
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([0.075, 0.17, 0.43, 0.72])
    series = [  # key, values, color, marker, label at the line end
        ("q_late", q_late, C["q"], "o", "question,\nlate layers"),
        ("store", store, C["word"], "^", "word-level\nstore"),
        ("q_early", q_early, BLUE_RAMP[0], "o", "question,\nearly layers"),
        ("nq_late", nq_late, C["nq"], "s", "no question,\nlate layers")]
    # per-model connector: late answer position vs the model's own store
    ax.vlines(x, np.minimum(store, q_late), np.maximum(store, q_late), color="#d5d3cc", lw=1.3,
              zorder=1)
    ends = {}
    for key, y, col, mk, lab in series:
        b, a = np.polyfit(x, y, 1)
        xx = np.array([x.min(), x.max()])
        ax.plot(xx, a + b * xx, color=col, lw=1.6, zorder=2)
        ax.scatter(x, y, s=24 if mk != "^" else 28, marker=mk, color=col, **RING)
        ends[key] = a + b * x.max()
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.text(x.min() - 0.06, -0.012, "word-only floor", fontsize=7.2, color=MUTED, va="top")
    ax.set_xlim(x.min() - 0.08, x.max() + 0.08)
    ax.set_ylim(-0.065, 0.49)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(["0", "0.1", "0.2", "0.3", "0.4"])
    size_axis(ax)
    ax.set_ylabel("probe accuracy above the floor", fontsize=8)
    # direct labels at the right end of each line
    for key, y, col, mk, lab in series:
        ax.text(x.max() + 0.13, ends[key], lab, fontsize=7.4, color=INK, va="center", ha="left",
                linespacing=1.05, clip_on=False)
        ax.plot([x.max() + 0.03, x.max() + 0.11], [ends[key]] * 2, color=col, lw=1.6,
                clip_on=False, solid_capstyle="butt")
    fig.text(0.0, 0.955, "last_letter, zero shots, one point per model",
             fontsize=8.5, fontweight="bold", color=INK, va="bottom")

    # ---------------- design key: two of the three prompts, probed positions marked (no data)
    kx0, kw = 0.655, 0.345
    k = fig.add_axes([kx0, 0.0, kw, 1.0])
    k.axis("off")
    fs = 7.4
    cw = fs * 0.602 / 72                                   # DejaVu Sans Mono advance width (in)
    k.set_xlim(0, kw * FW / cw)
    H = FH
    k.set_ylim(0, H)
    k.text(0, 0.955 * H, "Where the probe reads", fontsize=8.5, fontweight="bold", color=INK,
           va="bottom")
    ys = [H - 0.56, H - 1.04]
    for (arm, _, _), y in zip(PROMPTS, ys):
        k.text(0, y + 0.2, arm + (" prompt" if arm == "question" else " sentence (asks nothing)"),
               ha="left", va="center", fontsize=7.6, color=MUTED)
    for xm, y, kk in prompt_key(k, 0.0, ys, fs, 0.75, rows=PROMPTS[:2]):
        site_marker(k, xm, y - 0.165, kk)
    leg = [("word", "word tokens, no question: the word-level store"),
           ("q", "answer position (final token), question:\nlate layers (≥ 0.5 depth), early (≤ 0.25)"),
           ("nq", "answer position, no question: late layers")]
    y0 = ys[1] - 0.55
    for i, (kk, txt) in enumerate(leg):
        y = y0 - [0, 0.27, 0.54][i]
        site_marker(k, 1.0, y, kk)
        if kk == "q":
            site_marker(k, 2.9, y, kk, color=BLUE_RAMP[0])
        k.text(4.5 if kk == "q" else 2.6, y, txt, fontsize=7.4, color=INK, va="center",
               linespacing=1.2)
    save(fig, "fig_overview")


def draw_slopes(fig, L, R, B, T, FH):
    """Growth of the question effect with scale, by depth band (former Figure 1c): slope per decade
    of the per-model (question - no question) difference at the answer position, with 95% family-
    resampling CI. All layers: S.E5 '<task>|s<sh>' question_minus_neutral (as Table tab_interaction);
    late and early bands: F.F1 's<sh>' '<task>|late|q_minus_nq', '<task>|early|q_minus_nq'."""
    from matplotlib.lines import Line2D
    tasks = [t for t, _ in fu.PROMPT_SENSITIVE]
    bands = [("all layers", "all", INK2), ("late layers (≥ 0.5 depth)", "late", C["q"]),
             ("early layers (≤ 0.25 depth)", "early", MUTED2)]
    g = 0.02
    w = (R - L - 2 * g) / 3
    ys = np.arange(len(tasks))[::-1].astype(float)
    for j, (title, band, col) in enumerate(bands):
        ax = fig.add_axes([L + j * (w + g), B, w, T - B])
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        for t, y in zip(tasks, ys):
            for sh, dy in ((0, 0.16), (3, -0.16)):
                d = (S["E5"][f"{t}|s{sh}"]["question_minus_neutral"] if band == "all"
                     else F["F1"][f"s{sh}"][f"{t}|{band}|q_minus_nq"])
                lo, hi = d["family_boot_ci95"]
                ax.plot([lo, hi], [y + dy] * 2, color=col, lw=1.2, zorder=2, solid_capstyle="round")
                hollow = sh == 0
                ax.plot(d["slope_per_decade"], y + dy, marker="o", ms=4.8, ls="none", zorder=3,
                        mfc="white" if hollow else col, mec=col if hollow else "white",
                        mew=1.0 if hollow else 0.5)
        ax.set_xlim(-0.12, 0.32)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_xticklabels(["0", "0.1", "0.2", "0.3"])
        ax.tick_params(axis="x", pad=2)
        ax.set_ylim(-0.55, len(tasks) - 0.45)
        ax.set_title(title, fontsize=8, color=INK, pad=3)
        task_rows(ax, tasks, ys, labels=j == 0)
    fig.text((L + R) / 2, B - 0.24 / FH, "slope per decade of parameters, question − no question "
             "(95% family interval)", ha="center", fontsize=8, color=INK, va="top")
    h = [Line2D([], [], marker="o", ls="none", ms=4.8, mfc="white", mec=INK2, mew=1.0),
         Line2D([], [], marker="o", ls="none", ms=4.8, mfc=INK2, mec="white", mew=0.5)]
    fig.legend(h, ["0 shots", "3 shots"], loc="center left", ncol=2,
               bbox_to_anchor=(0.7, 0.43), fontsize=7.4, handletextpad=0.2,
               columnspacing=1.0)


# ============================================================================ figure: store
STORE_TASKS = ["first_letter", "last_letter", "most_frequent_letter", "word_length",
               "count_distinct_lm"]
EMB = [("last_letter", "orig", "word_last", "last token"),
       ("most_frequent_letter", "cvs", "word_mean", "span mean"),
       ("first_letter", "orig", "word_mean", "span mean")]


def shot_pair(ax, v0, v3, y, mk, col, ms, hollow_both=False):
    """One 0-shot -> 3-shot pair: hollow marker at 0 shots, filled at 3, joined by a thin line."""
    ax.plot([v0, v3], [y, y], color=col, lw=1.0, alpha=0.45, zorder=2, solid_capstyle="butt")
    ax.plot(v0, y, marker=mk, ms=ms, mfc="white", mec=col, mew=1.0, ls="none", zorder=3)
    ax.plot(v3, y, marker=mk, ms=ms, mfc=col, mec="white", mew=0.5, ls="none", zorder=4)


def fig_store():
    """(a) S.E1 '<task>|s0' and '|s3' mean_above_floor for word (last token, span mean) and answer
    (no question, question) positions, as Table tab_store, one 0 -> 3 shot pair per cell; (b) layer-0
    part of the word-level store (embedding_parts; the printed share equals numbers.json
    embedding_share), on the same task rows."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(7.2, 2.2))
    cells = [("word_nq_last", "word tokens, last token", "^", C["word"], 5.6),
             ("word_nq", "word tokens, span mean", "v", C["word"], 5.6),
             ("ans_nq", "answer position, no question", "s", C["nq"], 4.8),
             ("ans_q", "answer position, question", "o", C["q"], 5.0)]
    offs = [0.3, 0.1, -0.1, -0.3]
    ys = np.arange(len(STORE_TASKS))[::-1].astype(float) * 1.05
    bot, hgt = 0.16, 0.585
    ax = fig.add_axes([0.205, bot, 0.40, hgt])
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    for t, y in zip(STORE_TASKS, ys):
        for (k, lab, mk, col, ms), dy in zip(cells, offs):
            v0, v3 = (S["E1"][f"{t}|s{sh}"][k]["mean_above_floor"] for sh in (0, 3))
            shot_pair(ax, v0, v3, y + dy, mk, col, ms)
    ax.set_xlim(-0.1, 0.6)
    ax.set_xticks([0, 0.2, 0.4, 0.6])
    ax.set_xticklabels(["0", "0.2", "0.4", "0.6"])
    ax.set_ylim(ys.min() - 0.55, ys.max() + 0.55)
    ax.set_xlabel("accuracy above floor, mean over 16 models", fontsize=8)
    task_rows(ax, STORE_TASKS, ys)
    for y in (ys[:-1] + ys[1:]) / 2:
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
    h = [Line2D([], [], marker=mk, ls="none", ms=ms, mfc=col, mec="white", mew=0.5)
         for _, _, mk, col, ms in cells]
    h += [Line2D([], [], marker="o", ls="none", ms=5.0, mfc="white", mec=INK2, mew=1.0),
          Line2D([], [], marker="o", ls="none", ms=5.0, mfc=INK2, mec="white", mew=0.5)]
    lab = [c[1] for c in cells] + ["0 shots (hollow)", "3 shots (filled)"]
    fig.legend(h, lab, loc="lower left", ncol=3, bbox_to_anchor=(0.0, 0.765),
               fontsize=7.4, handletextpad=0.2, columnspacing=1.0, labelspacing=0.3)
    fig.text(0.0, 0.97, "(a)  mean over the 16 models, by position and prompt", fontsize=8.5,
             fontweight="bold", color=INK, va="center")
    # (b) layer-0 part of the word-level store, on the same task rows
    axb = fig.add_axes([0.66, bot, 0.27, hgt], sharey=ax)
    emb = {t: (ts, read, rname) for t, ts, read, rname in EMB}
    for t, y in zip(STORE_TASKS, ys):
        if t not in emb:
            continue
        ts, read, rname = emb[t]
        for sh, dy in ((0, 0.19), (3, -0.19)):
            l0, full = embedding_parts(t, ts, sh, read)
            axb.barh(y + dy, l0, height=0.34, color=GREEN_DARK, edgecolor="white", linewidth=1.0)
            axb.barh(y + dy, full - l0, left=l0, height=0.34, color=GREEN_LIGHT,
                     edgecolor="white", linewidth=1.0)
            axb.text(full + 0.012, y + dy, f"{100 * l0 / full:.0f}%", va="center", fontsize=7.2,
                     color=INK)
            axb.text(-0.012, y + dy, f"{sh}", va="center", ha="right", fontsize=7.2, color=MUTED)
    axb.tick_params(axis="y", left=False, labelleft=False)
    axb.grid(axis="y", visible=False)
    axb.set_xlim(0, 0.56)
    axb.set_xticks([0, 0.2, 0.4])
    axb.set_xticklabels(["0", "0.2", "0.4"])
    axb.set_xlabel("word-level store above floor", fontsize=8)
    for y in (ys[:-1] + ys[1:]) / 2:
        axb.axhline(y, color=GRID, lw=0.6, zorder=0)
    axb.legend([Patch(color=GREEN_DARK), Patch(color=GREEN_LIGHT)],
               ["layer 0 (embedding)", "later layers"], loc="center left",
               bbox_to_anchor=(0.0, (np.mean(ys[3:]) - ax.get_ylim()[0]) / np.ptp(ax.get_ylim())),
               fontsize=7.4, handlelength=1.0, handletextpad=0.4, labelspacing=0.5,
               title="bars: 0 and 3 shots; store read is\nlast token for last_letter, else span mean",
               title_fontsize=7.2,
               alignment="left")
    fig.text(0.635, 0.97, "(b)  part of the store at layer 0", fontsize=8.5, fontweight="bold",
             color=INK, va="center")
    save(fig, "fig_store")


# ============================================================================ figure: retained
RET_TASKS = ["extraction", "last_letter", "most_frequent_letter", "letter_before", "word_length",
             "counting_lm", "count_distinct_lm", "first_letter", "contains_letter"]


def fig_retained():
    """Retained fraction, as Table tab_retained (S.E2 '<task>|s0' and '<task>|s3'): neutral carrier
    against the majority baseline and against the floor, continuation carrier against the floor;
    each value is one 0 -> 3 shot pair (hollow -> filled), as in fig_store."""
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(7.2, 2.25))
    vals = [("neutral_retained_above_majority", "neutral carrier, against majority", "s", MUTED, 4.6),
            ("neutral_retained_above_floor", "neutral carrier, against floor", "s", C["nq"], 4.8),
            ("continuation_retained_above_floor", "continuation carrier, against floor", "D", C["nq"],
             4.2)]
    offs = [0.26, 0.0, -0.26]
    ys = np.arange(len(RET_TASKS))[::-1].astype(float) * 1.05
    ys[5:] -= 0.45                                        # gap after the question-driven group
    ax = fig.add_axes([0.205, 0.135, 0.785, 0.715])
    for xv in (0, 1):
        ax.axvline(xv, color=MUTED, lw=0.8, zorder=1)
    for t, y in zip(RET_TASKS, ys):
        for (k, lab, mk, col, ms), dy in zip(vals, offs):
            v0, v3 = (S["E2"][f"{t}|s{sh}"].get(k) for sh in (0, 3))
            if v0 is None or v3 is None:                  # letter_before: no floor-based value
                assert v0 is None and v3 is None
                continue
            shot_pair(ax, v0, v3, y + dy, mk, col, ms)
    for y in (ys[:-1] + ys[1:]) / 2:
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)
    ax.set_xlim(-0.1, 1.27)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(ys.min() - 0.5, ys.max() + 0.5)
    ax.set_xlabel("retained fraction of the question arm's signal at the answer position", fontsize=8)
    task_rows(ax, RET_TASKS, ys)
    h = [Line2D([], [], marker=mk, ls="none", ms=ms, mfc=col, mec="white", mew=0.5)
         for _, _, mk, col, ms in vals]
    h.insert(1, Line2D([], [], marker="o", ls="none", ms=5.0, mfc="white", mec=INK2, mew=1.0))
    h.insert(3, Line2D([], [], marker="o", ls="none", ms=5.0, mfc=INK2, mec="white", mew=0.5))
    lab = [vals[0][1], "0 shots (hollow)", vals[1][1], "3 shots (filled)", vals[2][1]]
    fig.legend(h, lab, loc="lower left", ncol=3, bbox_to_anchor=(0.2, 0.855), fontsize=7.4,
               handletextpad=0.2, columnspacing=1.4, labelspacing=0.35)
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
