#!/usr/bin/env python3
"""WORD-POSITION vs ANSWER-POSITION probe (experiment E1), also the question-deletion control.

Motivation. A probe at the ANSWER position alone peaks at 0.84-0.95 relative depth, i.e. where
the model assembles its next token. "The answer
position encodes the last letter when the prompt asks for it, and not after a neutral sentence"
is close to expected. What makes it informative is a PROMPT-INDEPENDENT baseline: is the
property linearly present at the QUERY WORD'S OWN TOKENS, before any question can have acted?

For each (model, task set, shots) this script builds the SAME examples as the main grid (same
seed, words, labels, folds) and runs three prompt ARMS through the model:

  question      the task prompt (identical to run_grid's query arm)
  neutral       `The word "cat" appears in this sentence.`     (run_noquery carrier)
  continuation  `Consider the word "cat". The answer is`       (run_noquery carrier)

and probes, with ONE estimator (held-out-word GroupKFold, 12 evenly spaced layers, held-out
layer selection, cold-start folds):

  answer position   every arm, every task in the set            -> question-deletion control
  word position     question + neutral arms, pure-word tasks     -> the prompt-independent store
                    (a) last token of the query word
                    (b) mean over all tokens overlapping the query word

Reading. In the NEUTRAL carrier the word tokens are preceded only by `The word "` (causal
attention), so word@neutral is the cleanest prompt-independent measure available. The target
claim of the v4 paper is:
    word@neutral high (above the nuisance floor) and flat-or-rising in scale,
    answer@neutral low, answer@question high and rising in scale
  => orthography is STORED at the word regardless of the prompt and REACHES the answer position
     on demand, with routing that strengthens with scale.
If word@neutral is itself low, the "store" reading fails and the paper stays a methods paper.

Word-position probes are restricted to the PURE-WORD tasks (label is a function of the word
alone): for term tasks (extraction, letter_before, counting, contains_letter) the query term is
not visible at the word position in every arm, so the label is not identifiable there.

Usage:
  uv run python scripts/run_wordpos.py --model Qwen/Qwen3-8B --shots 0,3 --sets orig,cvs,lm \
      --words-file src/stimulus_words.txt --skip-existing
Output: outputs/v4/wordpos/wordpos_{set}_{model}_s{shots}.json
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
import run_noquery as nq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARMS = ("question", "neutral", "continuation")
WORD_ARMS = ("question", "neutral")          # continuation's word prefix ~= neutral's


# F3 STEM-ORDER mode: the same property asked with the question BEFORE vs AFTER the word, so the
# only difference at the word tokens is whether the question has been read yet.
STEM_PROPS = {"first_letter": "first letter", "last_letter": "last letter",
              "most_frequent_letter": "most frequent letter",
              "word_length": "number of letters"}


def stem_order_prompt(order, task, w, tokenizer, ans):
    prop = STEM_PROPS[task]
    stem = (f'The {prop} of the word "{w}" is' if order == "qfirst"
            else f'In the word "{w}", the {prop} is')
    return s1._wrap(stem, ans, [], tokenizer)   # trailing-space rule for digit answers


def is_pure(task):
    return s1.base_task(task) in nq.PURE_WORD_TASKS


def out_path(out_dir, model, taskset, shots):
    return f"{out_dir}/wordpos_{taskset}_{s1.model_tag(model)}_s{shots}.json"


def probe_positions(H_by_pos, data, task_list, ex_words, device, layers, kw):
    """{position: {task: probe_arm result}} for the positions present in H_by_pos."""
    tasks, targets = data["tasks"], data["target_ids"]
    res = {}
    for pos, H in H_by_pos.items():
        res[pos] = {}
        for t in task_list:
            m = tasks == t
            if not m.any():
                continue
            r = s1.probe_arm({L: H[L][m] for L in layers if L in H}, targets[m], ex_words[m],
                             device, layers, **kw)
            if r:
                res[pos][t] = r
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", required=True)
    ap.add_argument("--shots", default="0,3", help="comma-separated shot counts")
    ap.add_argument("--sets", default="orig,cvs,lm")
    ap.add_argument("--n", type=int, default=800, help="examples per task (match the main grid)")
    ap.add_argument("--words-file", default="src/stimulus_words.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--probe-layers", type=int, default=12,
                    help="fixed NUMBER of evenly spaced probed layers (depth-comparable)")
    ap.add_argument("--group-folds", type=int, default=3)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="word-level bootstrap resamples for 95%% CIs on above-majority (0=off)")
    ap.add_argument("--warm-start", action="store_true",
                    help="warm-start CV folds (v3 behaviour; v4 default is cold start)")
    ap.add_argument("--out-dir", default="outputs/v4/wordpos")
    ap.add_argument("--no-remote-code", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--stem-order", action="store_true",
                    help="F3: arms qfirst / wfirst / neutral on the STEM_PROPS tasks, shots=0 only")
    args = ap.parse_args()
    if args.stem_order:
        if args.shots != "0":
            raise SystemExit("--stem-order is defined for --shots 0 only")
        if args.out_dir == "outputs/v4/wordpos":
            args.out_dir = "outputs/v4/stemorder"

    shots_list = [int(s) for s in args.shots.split(",") if s != ""]
    sets_list = [s for s in args.sets.split(",") if s]
    todo = [(sh, ts) for sh in shots_list for ts in sets_list
            if not (args.skip_existing and Path(out_path(args.out_dir, args.model, ts, sh)).exists())]
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
    layers = s1.even_layers(n_hidden, args.probe_layers)
    kw = {"folds": args.group_folds, "warm_start": args.warm_start, "heldout": True,
          "n_boot": args.bootstrap}
    logger.info("word pool %d; layers %s; estimator %s", len(words), layers, kw)

    for sh, ts in todo:
        task_list = s1.TASK_SETS[ts]
        if args.stem_order:
            task_list = [t for t in task_list if t in STEM_PROPS]
            if not task_list:
                continue
        pure = [t for t in task_list if is_pure(t)]
        # SAME examples as the main grid (built from the full set so the RNG stream matches)
        examples = s1.build_examples(words, args.n, args.seed, shots=sh,
                                     tokenizer=tokenizer, tasks=s1.TASK_SETS[ts])
        examples = [e for e in examples if e["task"] in task_list]
        ex_words = np.array([e["word"] for e in examples])
        neutral = nq.build_carrier_examples(examples, words, "neutral", sh, args.seed)
        if args.stem_order:
            arm_names, word_arms = ("qfirst", "wfirst", "neutral"), ("qfirst", "wfirst", "neutral")
            prompts = {o: [dict(e, prompt=stem_order_prompt(o, e["task"], e["word"], tokenizer,
                                                            e["answer"])) for e in examples]
                       for o in ("qfirst", "wfirst")}
            prompts["neutral"] = neutral
        else:
            arm_names, word_arms = ARMS, WORD_ARMS
            prompts = {"question": examples, "neutral": neutral,
                       "continuation": nq.build_carrier_examples(examples, words, "continuation",
                                                                 sh, args.seed)}
        arms = {}
        for arm in arm_names:
            want_word = arm in word_arms and bool(pure)
            data = s1.extract(model, tokenizer, prompts[arm], args.device,
                              batch_size=args.batch_size, keep_layers=layers,
                              want_word=want_word, word_mean=want_word)
            res = probe_positions({"answer": data["ans_h"]}, data, task_list, ex_words,
                                  args.device, layers, kw)
            if args.stem_order and arm != "neutral":   # model's own forced-choice readout
                res["readout"] = {}
                for t in task_list:
                    mt = data["tasks"] == t
                    sp = np.array(s1.answer_space_ids(tokenizer, t))
                    pred = sp[data["final_logits"][mt][:, sp].argmax(1)]
                    res["readout"][t] = round(float(np.mean(pred == data["target_ids"][mt])), 4)
            if want_word:
                res.update(probe_positions({"word_last": data["word_h"],
                                            "word_mean": data["word_mean_h"]},
                                           data, pure, ex_words, args.device, layers, kw))
            arms[arm] = res
            del data
            logger.info("s%d %s arm=%s done", sh, ts, arm)

        # precomputed contrasts on the above-majority scale (peak and held-out-selected)
        summary = {}
        for t in ([] if args.stem_order else task_list):
            row = {"pure_word_function": t in pure}
            for key in ("above_majority", "above_majority_heldout_sel"):
                g = lambda arm, pos: (arms.get(arm, {}).get(pos, {}).get(t) or {}).get(key)
                q, nqa, cont = g("question", "answer"), g("neutral", "answer"), \
                    g("continuation", "answer")
                wn, wq = g("neutral", "word_mean"), g("question", "word_mean")
                sfx = "" if key == "above_majority" else "_hsel"
                row.update({f"ans_question{sfx}": q, f"ans_neutral{sfx}": nqa,
                            f"ans_continuation{sfx}": cont,
                            f"word_neutral{sfx}": wn, f"word_question{sfx}": wq})
            summary[t] = row

        lm_meta = {t: dict(zip(("lengths", "answers"), s1.lm_design(t, words)))
                   for t in task_list if t in s1.LM_TASKS}
        p = Path(out_path(args.out_dir, args.model, ts, sh))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "arms": arms, "summary": summary,
            "meta": {"model": args.model, "shots": sh, "set": ts, "seed": args.seed,
                     "n_per_task": args.n, "words_file": args.words_file, "n_words": len(words),
                     "probe_layers": layers, "n_hidden": n_hidden, "group_folds": args.group_folds,
                     "estimator": {"cold_start": not args.warm_start, "layer_select": "heldout",
                                   "bootstrap": args.bootstrap},
                     "pure_word_tasks": pure, "lm_design": lm_meta,
                     "legacy_set_order": s1.LEGACY_SET_ORDER,
                     "stem_order": args.stem_order,
                     "examples": {arm: next(x["prompt"] for x in prompts[arm]
                                            if x["task"] == task_list[0]) for arm in arm_names}}},
                    indent=2))
        for t, r in summary.items():
            logger.info("%-20s ans@q=%s ans@nq=%s word@nq=%s word@q=%s", t,
                        r["ans_question"], r["ans_neutral"], r["word_neutral"], r["word_question"])
        logger.info("saved -> %s", p)


if __name__ == "__main__":
    main()
