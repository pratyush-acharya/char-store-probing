#!/usr/bin/env python3
"""CROSS-TASK CONTROL: is the question COMPUTING the property, or merely SELECTING it?

Motivation. The no-question control
(`scripts/run_noquery.py`) established that THE QUESTION IS NECESSARY for the character answer to
be linearly decodable at the answer position. Two mechanisms explain that equally well:

  (1) COMPUTATION ON DEMAND -- the model runs a small operation because it was asked, and leaves
      the result at the answer position. (This is what the paper currently asserts.)
  (2) SELECTION / ROUTING ON DEMAND -- the property is already present in a richer (non-linear or
      distributed) form that the neutral carrier does not expose linearly at that position, and
      the question merely SELECTS it into a linearly readable form.

The discriminating experiment: **probe for property A while the prompt asks about property B.**

  A stays decodable under a B-prompt  -> the question is doing SELECTION/routing, not computing A.
                                         "Computation on demand" is too strong; the property is
                                         available whenever *some* question puts the model in an
                                         answering posture.
  A collapses under a B-prompt        -> A is decodable only when A is asked. "Computed on demand"
                                         is earned.

Read the cross arm against BOTH ends:
    self_minus_cross   = acc(ask A) - acc(ask B)     large => task-specific computation
    cross_minus_noquery= acc(ask B) - acc(no question) large => generic elicitation/selection
A cross arm that sits at the no-question floor is the "computed" result; a cross arm that sits at
the self level is the "selected" result; anything in between quantifies the mixture.

--------------------------------------------------------------------------------------------
TASK PAIRING (the key judgement)

Probed labels A -- the DEFAULT set is the five PURE WORD FUNCTIONS:
    first_letter, last_letter, most_frequent_letter, word_length, count_distinct
For these the label is a function of the word alone, so it is IDENTICAL whether the prompt asks A
or B -- the matched-stimulus requirement is satisfied by construction. Two of the paper's four
"computed" tasks (last_letter, most_frequent_letter) live here, and they are the two with the
largest and least demo-attributable no-question deltas (+0.394 / +0.335, demo share 0.08).

Asked questions B -- the SAME five pure tasks (each A skips its own diagonal). All are questions
about the word alone, so the cross prompt is the ordinary B prompt with the word substituted: the
only thing that changes between the self and cross arms at shots=0 is the QUESTION TEXT.

EXCLUDED from B: contains_letter, counting, extraction, letter_before. They carry a query TERM,
which (a) adds a second sampled quantity to the prompt, and (b) can LEAK the A label outright --
`The 5th letter of the word "table" is` makes last_letter's answer the literal answer of B. A
cross arm inflated by leakage would masquerade as "selection" and would falsely refute the paper.

EXCLUDED from A by default: extraction and letter_before, the paper's other two "computed" tasks.
Their labels are defined only GIVEN a query term ("which position", "before which letter"). Under
a B-prompt that term is absent, so the label is not merely uncomputed but UNIDENTIFIABLE, and a
collapse would be trivial rather than evidence of computation. `--include-term-tasks` runs them
anyway, in a secondary form that prepends the term as a bare NOTE line before B's question
(`Note the letter "e".` / `Note the 3rd position.`), mirroring the term-carrying carriers in
run_noquery.py. Read those cells as exploratory; the prompt no longer differs only in the question.

Two built-in confound checks, both emitted:
  * `label_agreement` -- P(answer to B == label A) on the same word. Letter-vs-digit B choices make
    this structurally 0 and are therefore the cleanest pairs; a high value means the cross probe
    could be reading B's OWN answer rather than a retained A.
  * `taskb_sanity` -- the accuracy of decoding B's own label in the cross pass. If A collapses AND
    B is undecodable, the pass is broken, not informative.

Shot counts: 0 and 3. **0-shot is the clean contrast** -- with no demonstrations the self and cross
prompts differ ONLY in the question sentence. At shots=3 the demonstrations necessarily demonstrate
B, so the cross prompt differs in the demos too; treat shots=3 as a robustness arm.

Both (all) conditions are measured in ONE process on the SAME examples, with the same probe, the
same seed and the same GroupKFold-by-query-word folds, so no comparison is confounded by sampling
or estimator differences.

Usage:
  uv run python scripts/run_crosstask.py --model Qwen/Qwen3-8B --sets orig,cvs --shots 0 \
      --words-file src/stimulus_words.txt --skip-existing
Output: outputs/bigstim_crosstask/crosstask_{set}_{model}_s{shots}.json
"""
import argparse, json, logging, random, sys
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

# labels whose value is a pure function of the word -> identical under any question
PURE_WORD_TASKS = ["first_letter", "last_letter", "most_frequent_letter",
                   "word_length", "count_distinct"]
# labels defined only given a query term -> secondary, need the term carried into the B prompt
TERM_LABEL_TASKS = ["extraction", "letter_before"]
# questions we are willing to ask: pure-word questions only (no query term to sample or leak)
ASKABLE_TASKS = list(PURE_WORD_TASKS)
# answer type, used for the leak/agreement note and the _wrap spacing fallback
DIGIT_TASKS = {"word_length", "count_distinct", "counting"}


def note_line(kind, val):
    """Secondary arm only: carry a term-task's query term into a B prompt without asking about it."""
    if kind == "letter":
        return f'Note the letter "{val}".'
    if kind == "ordinal":
        return f"Note the {val} position."
    return None


def build_cross_examples(examples, words, task_b, shots, tokenizer, seed=42):
    """A B-question twin of each example: SAME word, SAME A label, different QUESTION.

    The prompt is built through exactly the same path build_examples() uses for task_b
    (`s1._demo` + `s1._stem` + `s1._wrap`), so the cross prompt is format-identical to a real
    task-B prompt. `answer` stays the A label (that is what `extract` turns into target_ids and
    what we probe); `answer_b` is B's own label, kept for the leak check and the sanity probe.
    Examples are never dropped -- if a word is unsuitable as a task-B *item* (a
    most_frequent_letter tie, a >9-letter word_length), the B QUESTION is still well formed, so we
    keep the prompt and record answer_b=None."""
    rng = random.Random(f"{seed}|{task_b}|{shots}")
    filler = "5" if task_b in DIGIT_TASKS else "a"   # only sets _wrap's trailing-space decision
    out = []
    for e in examples:
        w = e["word"]
        r = s1._sample_task(task_b, w, rng)
        if r is None:
            stem, ans_b = s1._stem(task_b, w), None
        else:
            stem, ans_b = r
        demos = [s1._demo(task_b, words, rng, w) for _ in range(shots)]
        prompt = s1._wrap(stem, ans_b if ans_b is not None else filler, demos, tokenizer)
        if e["task"] in TERM_LABEL_TASKS:            # secondary arm: carry A's term as a NOTE
            kind, val = nq.query_term(e["task"], e["prompt"])
            nl = note_line(kind, val)
            if nl:
                prompt = nl + "\n" + prompt
        out.append(dict(task=e["task"], word=w, answer=e["answer"], answer_b=ans_b,
                        asked=task_b, prompt=prompt))
    return out


def probe_condition(data, task_list, ex_words, device, probe_layers, folds=3, warm_start=True,
                    targets=None, valid=None, heldout=False):
    """Held-out-word probe at every probed layer; peak over layers. Identical estimator, folds and
    grouping to run_noquery.probe_condition / the main grid (s1.probe_acc_grouped).

    `targets` overrides the probed label (used for the task-B sanity probe); `valid` masks out
    rows with no defined label. `heldout` / `warm_start=False` are the v4 estimator fixes
    (see s1.probe_arm); the published fields are unchanged."""
    tasks = data["tasks"]
    targets = data["target_ids"] if targets is None else np.asarray(targets)
    ex_words = np.asarray(ex_words)
    res = {}
    for t in task_list:
        m = tasks == t
        if valid is not None:
            m = m & np.asarray(valid)
        if m.sum() == 0:
            continue
        H = {L: data["ans_h"][L][m] for L in probe_layers if L in data["ans_h"]}
        r = s1.probe_arm(H, targets[m], ex_words[m], device, probe_layers, folds=folds,
                         warm_start=warm_start, heldout=heldout)
        if r:
            res[t] = r
    return res


def out_path(model, taskset, shots, tag="", out_dir="outputs/bigstim_crosstask"):
    # `tag` keeps a secondary arm (e.g. --include-term-tasks) from overwriting, or being silently
    # skipped by --skip-existing against, the primary run's outputs: the filename is otherwise
    # identical for every flag combination.
    suffix = f"_{tag}" if tag else ""
    return (f"{out_dir}/crosstask_{taskset}_{s1.model_tag(model)}"
            f"_s{shots}{suffix}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", required=True)
    ap.add_argument("--shots", type=int, default=0,
                    help="0 = the clean contrast (prompts differ only in the question)")
    ap.add_argument("--sets", default="orig,cvs")
    ap.add_argument("--a-tasks", default="", help="comma-sep probed labels (default: the pure five "
                                                  "present in the set)")
    ap.add_argument("--b-tasks", default=",".join(ASKABLE_TASKS),
                    help="comma-sep asked questions (pure-word tasks only)")
    ap.add_argument("--include-term-tasks", action="store_true",
                    help="SECONDARY: also probe extraction/letter_before, carrying their query "
                         "term into the B prompt as a NOTE line (see module docstring)")
    ap.add_argument("--carrier", default="neutral", choices=list(nq.CARRIERS),
                    help="carrier for the no-question reference arm")
    ap.add_argument("--no-noquery-arm", action="store_true",
                    help="skip the no-question reference pass (then cross_minus_noquery is absent "
                         "and must be read against outputs/bigstim_noquery/)")
    ap.add_argument("--no-sanity-probe", action="store_true",
                    help="skip decoding task B's own label in the cross pass")
    ap.add_argument("--n", type=int, default=800, help="examples per task (match the main grid)")
    ap.add_argument("--words-file", default="src/stimulus_words.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--probe-layers", type=int, default=12,
                    help="fixed NUMBER of evenly-spaced probed layers (depth-comparable)")
    ap.add_argument("--group-folds", type=int, default=3)
    ap.add_argument("--no-remote-code", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--out-tag", default="",
                    help="suffix for the output filename; use for secondary arms (e.g. "
                         "--include-term-tasks --out-tag term) so they neither overwrite nor are "
                         "skipped against the primary run's outputs")
    # --- v4 estimator fixes (off by default -> published behaviour unchanged) ---
    ap.add_argument("--cold-start", action="store_true", help="v4: no warm-start across CV folds")
    ap.add_argument("--layer-select", default="max", choices=["max", "heldout"],
                    help="v4: 'heldout' also reports a fold-held-out layer choice")
    ap.add_argument("--out-dir", default="outputs/bigstim_crosstask",
                    help="v4: output directory (keeps the published crosstask files intact)")
    ap.add_argument("--only-cont-arm", action="store_true",
                    help="review R3: run ONLY the continuation carrier on this script's exact examples "
                         "and write <result>_cont.json beside the existing result")
    args = ap.parse_args()
    pc_kw = {"folds": args.group_folds, "warm_start": not args.cold_start,
             "heldout": args.layer_select == "heldout"}

    b_tasks = [t for t in args.b_tasks.split(",") if t]
    bad = [t for t in b_tasks if t not in ASKABLE_TASKS]
    if bad:
        raise SystemExit(f"--b-tasks must be pure-word tasks {ASKABLE_TASKS}; got {bad}")
    a_filter = [t for t in args.a_tasks.split(",") if t]

    sets_list = [s for s in args.sets.split(",") if s]
    cont_path = lambda ts: out_path(args.model, ts, args.shots, args.out_tag,
                                    args.out_dir).replace(".json", "_cont.json")
    todo = [ts for ts in sets_list
            if not (args.skip_existing
                    and Path(cont_path(ts) if args.only_cont_arm else
                             out_path(args.model, ts, args.shots, args.out_tag,
                                      args.out_dir)).exists())]
    if not todo:
        logger.info("skip (all outputs exist): %s", args.model)
        return

    config = ExperimentConfig()
    config.model.model_name = args.model
    apply_env_overrides(config)
    if args.no_remote_code:
        config.model.trust_remote_code = False

    logger.info("Loading %s on %s (ONCE)", args.model, args.device)
    model = CausalLM(config.model).to(args.device)
    model.eval()
    tokenizer = load_tokenizer(config.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    words = s1.load_word_pool(args.words_file)
    n_hidden = model.base_model.config.num_hidden_layers + 1
    probe_layers = s1.even_layers(n_hidden, args.probe_layers)
    logger.info("word pool: %d; probe layers: %s", len(words), probe_layers)

    allowed_a = PURE_WORD_TASKS + (TERM_LABEL_TASKS if args.include_term_tasks else [])
    for ts in todo:
        task_list = [t for t in s1.TASK_SETS[ts] if t in allowed_a]
        if a_filter:
            task_list = [t for t in task_list if t in a_filter]
        if not task_list:
            logger.info("no probeable labels in set %s; skipping", ts)
            continue
        # the SAME construction the main grid / no-query control uses (same seed/words/targets)
        examples = s1.build_examples(words, args.n, args.seed, shots=args.shots,
                                     tokenizer=tokenizer, tasks=task_list)
        ex_words = np.array([e["word"] for e in examples])

        if args.only_cont_arm:   # review R3: continuation floor on the SAME draw as the cross arms
            cex = nq.build_carrier_examples(examples, words, "continuation", args.shots, args.seed)
            d = s1.extract(model, tokenizer, cex, args.device, batch_size=args.batch_size,
                           keep_layers=probe_layers, want_word=False)
            cont = probe_condition(d, task_list, ex_words, args.device, probe_layers, **pc_kw)
            p = Path(cont_path(ts))
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"continuation": cont, "meta": {
                "model": args.model, "shots": args.shots, "set": ts, "a_tasks": task_list,
                "out_tag": args.out_tag, "example": cex[0]["prompt"] if cex else None,
                "probe_layers": probe_layers, "estimator": {"cold_start": args.cold_start,
                                                            "layer_select": args.layer_select}}},
                indent=2))
            logger.info("saved -> %s", p)
            continue

        # --- SELF arm: ask A (identical to the main grid's query arm) ---
        data = s1.extract(model, tokenizer, examples, args.device, batch_size=args.batch_size,
                          keep_layers=probe_layers, want_word=False)
        self_arm = probe_condition(data, task_list, ex_words, args.device, probe_layers,
                                   **pc_kw)
        a_targets = data["target_ids"].copy()
        del data

        # --- NO-QUESTION reference arm (same carrier as run_noquery) ---
        noquery = {}
        if not args.no_noquery_arm:
            carrier_ex = nq.build_carrier_examples(examples, words, args.carrier, args.shots,
                                                   args.seed)
            d = s1.extract(model, tokenizer, carrier_ex, args.device, batch_size=args.batch_size,
                           keep_layers=probe_layers, want_word=False)
            noquery = probe_condition(d, task_list, ex_words, args.device, probe_layers,
                                      **pc_kw)
            del d

        # --- CROSS arms: ask B, probe A ---
        cross, sanity, agree, examples_b = {}, {}, {}, {}
        for tb in b_tasks:
            if all(t == tb for t in task_list):
                continue                      # nothing but the diagonal in this set
            cex = build_cross_examples(examples, words, tb, args.shots, tokenizer, args.seed)
            d = s1.extract(model, tokenizer, cex, args.device, batch_size=args.batch_size,
                           keep_layers=probe_layers, want_word=False)
            # matched-stimulus assertion: same words, same A labels, same folds as the self arm
            assert all(a["word"] == b["word"] for a, b in zip(examples, cex))
            assert np.array_equal(d["target_ids"], a_targets)
            probe_a = [t for t in task_list if t != tb]
            cross[tb] = probe_condition(d, probe_a, ex_words, args.device, probe_layers,
                                        **pc_kw)
            if not args.no_sanity_probe:
                valid = np.array([e["answer_b"] is not None for e in cex])
                tb_ids = np.array([s1.first_content_token(tokenizer, e["answer_b"])
                                   if e["answer_b"] is not None else -1 for e in cex])
                sanity[tb] = probe_condition(d, probe_a, ex_words, args.device, probe_layers,
                                             targets=tb_ids, valid=valid, **pc_kw)
            agree[tb] = {t: round(float(np.mean([e["answer_b"] == e["answer"] for e in cex
                                                 if e["task"] == t])), 4) for t in probe_a}
            examples_b[tb] = cex[0]["prompt"] if cex else None
            del d

        # --- deltas: the interpretation, precomputed ---
        delta = {}
        for tb, res in cross.items():
            for t, r in res.items():
                s = self_arm.get(t)
                if s is None:
                    continue
                nqr = noquery.get(t)
                d = {"probe_label": t, "asked": tb,
                     "acc_self": s["acc_peak"], "acc_cross": r["acc_peak"],
                     "majority": s["majority"],
                     "above_majority_self": s["above_majority"],
                     "above_majority_cross": r["above_majority"],
                     "self_minus_cross": round(s["acc_peak"] - r["acc_peak"], 4),
                     "above_majority_self_minus_cross": round(s["above_majority"]
                                                              - r["above_majority"], 4),
                     "label_agreement": agree.get(tb, {}).get(t),
                     "taskb_sanity_acc": (sanity.get(tb, {}).get(t) or {}).get("acc_peak"),
                     "secondary_term_task": t in TERM_LABEL_TASKS}
                if nqr is not None:
                    d.update({
                        "acc_noquery": nqr["acc_peak"],
                        "above_majority_noquery": nqr["above_majority"],
                        "cross_minus_noquery": round(r["acc_peak"] - nqr["acc_peak"], 4),
                        "above_majority_cross_minus_noquery": round(r["above_majority"]
                                                                    - nqr["above_majority"], 4),
                        # 0 => cross sits at the no-question floor (COMPUTED);
                        # 1 => cross sits at the ask-A level (SELECTED)
                        "selection_index": (round((r["above_majority"] - nqr["above_majority"])
                                                  / (s["above_majority"] - nqr["above_majority"]), 4)
                                            if abs(s["above_majority"] - nqr["above_majority"]) > 1e-6
                                            else None)})
                delta[f"{t}|{tb}"] = d

        p = Path(out_path(args.model, ts, args.shots, args.out_tag, args.out_dir))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "self": self_arm, "noquery": noquery, "cross": cross, "taskb_sanity": sanity,
            "delta": delta,
            "meta": {"model": args.model, "shots": args.shots, "set": ts,
                     "a_tasks": task_list, "b_tasks": b_tasks, "carrier": args.carrier,
                     "noquery_arm": not args.no_noquery_arm,
                     "include_term_tasks": args.include_term_tasks,
                     "out_tag": args.out_tag,
                     "estimator": {"cold_start": args.cold_start,
                                   "layer_select": args.layer_select},
                     "self_example": examples[0]["prompt"] if examples else None,
                     "cross_examples": examples_b,
                     "seed": args.seed, "n_per_task": args.n, "words_file": args.words_file,
                     "n_words": len(words), "probe_layers": probe_layers,
                     "group_folds": args.group_folds, "grouped": True,
                     "pure_word_tasks": PURE_WORD_TASKS,
                     "term_label_tasks": TERM_LABEL_TASKS}}, indent=2))
        for k, d in delta.items():
            logger.info("%-34s above-maj self=%.3f cross=%.3f nq=%s  self-cross=%+.3f  sel_idx=%s",
                        k, d["above_majority_self"], d["above_majority_cross"],
                        f"{d['above_majority_noquery']:.3f}" if "above_majority_noquery" in d
                        else "n/a", d["above_majority_self_minus_cross"],
                        d.get("selection_index"))
        logger.info("saved -> %s", p)


if __name__ == "__main__":
    main()
