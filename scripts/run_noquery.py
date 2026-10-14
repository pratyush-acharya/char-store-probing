#!/usr/bin/env python3
"""NO-QUERY CONTROL: is the character answer *computed and suppressed*, or just a static
lexical feature of having read the word?

Motivation. For `word_length`, `count_distinct`,
`most_frequent_letter`, `first_letter` and `last_letter` the answer is a PURE FUNCTION OF THE
WORD -- no query term appears in the prompt. So "the answer is decodable on held-out words"
is dangerously close to "word identity is encoded, and these targets are (learnable) functions
of word identity." That is a much weaker claim than "the model computed the answer and failed
to route it to the output."

The control: take the SAME examples (same words, same targets, same seed), put each word in a
CARRIER prompt that does NOT ask the question, extract at the last prompt token, and run the
IDENTICAL held-out-word probe for the IDENTICAL target. Then

    delta = acc_query - acc_noquery

  delta ~ 0  ->  the property is a STATIC LEXICAL FEATURE of reading the word. The task prompt
                 adds nothing; "present but not predicted" is not evidence of a suppressed
                 computation, and §4's framing must be weakened for that task.
  delta > 0  ->  the query prompt genuinely elicits task-specific structure; the "computed but
                 not read out" reading survives.

By default we measure BOTH conditions in the same process, on the same examples, with the same
probe and the same folds, so the delta is not confounded by sampling or estimator differences.

Carrier design (the one judgement call):
  neutral      : `The word "cat" appears in this sentence.`
                 Zero task cue of any kind. The extraction position is the sentence-final token.
                 This is the strictest reading of "a prompt that does not ask the question".
  continuation : `Consider the word "cat". The answer is`
                 Matches the FUNCTIONAL ROLE of the query condition's extraction position -- the
                 last token of a stem awaiting a continuation -- without naming which property is
                 wanted. Controls for the objection that `neutral` merely measures a
                 sentence-final token rather than an "about to answer" token.
Both are run by the deploy script; a conclusion should hold under both. For the four tasks whose
answer is NOT a pure function of the word (contains_letter, counting, extraction, letter_before)
the carrier also mentions the query term, so the target stays computable in principle; those are
SECONDARY -- the primary five are the pure-function tasks the review names.

Few-shot matching: `--shots k` prepends k `neutral` carrier sentences about OTHER words, so the
prompt has comparable length and the same "model has been reading words" context, while teaching
the task nothing (there are no answers in a carrier).

Usage:
  uv run python scripts/run_noquery.py --model Qwen/Qwen3-8B --carrier neutral \
      --sets orig,cvs --shots 3 --words-file src/stimulus_words.txt --skip-existing
Output: outputs/bigstim_noquery/noquery_{carrier}_{set}_{model}_s{shots}.json
"""
import argparse, json, logging, random, re, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import ExperimentConfig, apply_env_overrides
from src.model import CausalLM, load_tokenizer
import run_stage1 as s1

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CARRIERS = ("neutral", "continuation")
# tasks whose answer is a pure function of the word (no query term in the prompt) -- the review's
# target set, and the only ones where "static lexical feature" is the competing explanation
PURE_WORD_TASKS = {"first_letter", "last_letter", "most_frequent_letter",
                   "word_length", "count_distinct"}


def query_term(task, prompt):
    """Recover the query term from an example's prompt (the LAST line = the query stem).

    build_examples() does not record it, and we must not change its output schema, so we parse
    the stem formats in run_stage1._stem. Returns (kind, value): ('letter', c), ('ordinal', o)
    or (None, None) for the pure-function tasks."""
    line = prompt.split("\n")[-1]
    task = s1.base_task(task)          # counting_lm -> counting (same question, v4 E6)
    if task in ("contains_letter", "counting", "letter_before"):
        q = re.findall(r'"([^"]*)"', line)
        return ("letter", q[1]) if len(q) > 1 else (None, None)
    if task == "extraction":
        m = re.search(r"The (\w+) letter of", line)
        return ("ordinal", m.group(1)) if m else (None, None)
    return (None, None)


def carrier_line(style, w, kind=None, val=None):
    """One carrier sentence mentioning word `w` (and the query term, if the task has one)."""
    if style == "continuation":
        if kind == "letter":
            return f'Consider the word "{w}" and the letter "{val}". The answer is'
        if kind == "ordinal":
            return f'Consider the word "{w}" and the {val} position. The answer is'
        return f'Consider the word "{w}". The answer is'
    if kind == "letter":
        return f'The word "{w}" and the letter "{val}" appear in this sentence.'
    if kind == "ordinal":
        return f'The word "{w}" and the {val} position appear in this sentence.'
    return f'The word "{w}" appears in this sentence.'


def build_carrier_examples(examples, words, style, shots, seed=42):
    """A carrier-prompt twin of each example: SAME task, SAME word, SAME answer, no question."""
    rng = random.Random(seed)
    out = []
    for e in examples:
        kind, val = query_term(e["task"], e["prompt"])
        demos = []
        for _ in range(shots):
            w2 = rng.choice(words)
            for _ in range(10):
                if w2 != e["word"]:
                    break
                w2 = rng.choice(words)
            demos.append(carrier_line("neutral", w2))   # demos never carry task information
        stem = carrier_line(style, e["word"], kind, val)
        out.append(dict(task=e["task"], word=e["word"], answer=e["answer"],
                        prompt="\n".join(demos + [stem])))
    return out


def probe_condition(data, task_list, ex_words, device, probe_layers, folds=3, warm_start=True):
    """Held-out-word probe of the answer target at every probed layer; peak over layers.
    Identical estimator to the main grid (s1.probe_acc_grouped)."""
    tasks, targets = data["tasks"], data["target_ids"]
    ex_words = np.asarray(ex_words)
    res = {}
    for t in task_list:
        m = tasks == t
        if m.sum() == 0:
            continue
        by_layer, meta = {}, None
        for L in probe_layers:
            if L not in data["ans_h"]:
                continue
            r = s1.probe_acc_grouped(data["ans_h"][L][m], targets[m], ex_words[m], device,
                                     folds=folds, warm_start=warm_start)
            if r:
                by_layer[L] = round(r["acc"], 4)
                if meta is None or r["acc"] >= max(by_layer.values()):
                    meta = r
        if not by_layer or meta is None:
            continue
        peak_L = max(by_layer, key=by_layer.get)
        maj = meta.get("majority", meta["chance"])
        res[t] = {"acc_peak": by_layer[peak_L], "peak_layer": int(peak_L),
                  "majority": round(float(maj), 4),
                  "above_majority": round(by_layer[peak_L] - float(maj), 4),
                  "chance": round(meta["chance"], 3), "n_classes": meta["n_classes"],
                  "n_words": meta["n_words"], "by_layer": by_layer}
    return res


def out_path(model, taskset, shots, carrier):
    return (f"outputs/bigstim_noquery/noquery_{carrier}_{taskset}"
            f"_{s1.model_tag(model)}_s{shots}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", required=True)
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--sets", default="orig,cvs")
    ap.add_argument("--carrier", default="neutral", choices=list(CARRIERS))
    ap.add_argument("--n", type=int, default=800, help="examples per task (match the main grid)")
    ap.add_argument("--words-file", default="src/stimulus_words.txt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--probe-layers", type=int, default=12,
                    help="fixed NUMBER of evenly-spaced probed layers (depth-comparable)")
    ap.add_argument("--group-folds", type=int, default=3)
    ap.add_argument("--no-query-condition", action="store_true",
                    help="skip the matched query condition (halves the cost, but then the delta "
                         "must be read against the separately-run main grid)")
    ap.add_argument("--pure-only", action="store_true",
                    help="only the 5 tasks whose answer is a pure function of the word")
    ap.add_argument("--no-remote-code", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    sets_list = [s for s in args.sets.split(",") if s]
    todo = [ts for ts in sets_list
            if not (args.skip_existing and Path(out_path(args.model, ts, args.shots,
                                                         args.carrier)).exists())]
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

    for ts in todo:
        task_list = s1.TASK_SETS[ts]
        if args.pure_only:
            task_list = [t for t in task_list if t in PURE_WORD_TASKS]
            if not task_list:
                logger.info("no pure-word tasks in set %s; skipping", ts)
                continue
        # the SAME examples the main grid uses (same seed/words/targets)
        examples = s1.build_examples(words, args.n, args.seed, shots=args.shots,
                                     tokenizer=tokenizer, tasks=task_list)
        ex_words = np.array([e["word"] for e in examples])
        carrier_ex = build_carrier_examples(examples, words, args.carrier, args.shots, args.seed)

        data_nq = s1.extract(model, tokenizer, carrier_ex, args.device,
                             batch_size=args.batch_size, keep_layers=probe_layers, want_word=False)
        noquery = probe_condition(data_nq, task_list, ex_words, args.device, probe_layers,
                                  folds=args.group_folds)
        del data_nq

        query = {}
        if not args.no_query_condition:
            data_q = s1.extract(model, tokenizer, examples, args.device,
                                batch_size=args.batch_size, keep_layers=probe_layers,
                                want_word=False)
            query = probe_condition(data_q, task_list, ex_words, args.device, probe_layers,
                                    folds=args.group_folds)
            del data_q

        delta = {}
        for t in noquery:
            if t in query:
                delta[t] = {
                    "acc_query_minus_noquery": round(query[t]["acc_peak"] - noquery[t]["acc_peak"], 4),
                    "above_majority_query": query[t]["above_majority"],
                    "above_majority_noquery": noquery[t]["above_majority"],
                    "above_majority_delta": round(query[t]["above_majority"]
                                                  - noquery[t]["above_majority"], 4),
                    "pure_word_function": t in PURE_WORD_TASKS,
                }
        p = Path(out_path(args.model, ts, args.shots, args.carrier))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "noquery": noquery, "query": query, "delta": delta,
            "meta": {"model": args.model, "shots": args.shots, "set": ts, "carrier": args.carrier,
                     "carrier_example": carrier_ex[0]["prompt"] if carrier_ex else None,
                     "query_example": examples[0]["prompt"] if examples else None,
                     "seed": args.seed, "n_per_task": args.n, "words_file": args.words_file,
                     "n_words": len(words), "probe_layers": probe_layers,
                     "group_folds": args.group_folds, "grouped": True,
                     "pure_word_tasks": sorted(PURE_WORD_TASKS)}}, indent=2))
        for t, d in delta.items():
            logger.info("%-20s above-maj query=%.3f  noquery=%.3f  delta=%+.3f%s", t,
                        d["above_majority_query"], d["above_majority_noquery"],
                        d["above_majority_delta"], "  [pure]" if d["pure_word_function"] else "")
        logger.info("saved -> %s", p)


if __name__ == "__main__":
    main()
