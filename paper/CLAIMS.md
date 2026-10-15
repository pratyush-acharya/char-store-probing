# Claims ledger

The reference for what the paper may claim and how strongly. Any revision must keep every claim
below at its stated strength: **not stronger** (over-claiming) and **not weaker** (under-selling).
Baseline: git tag `paper-baseline-v1`. Numbers come from `outputs/v4/analysis/*.json` and
`paper/tables/numbers.json`; `scripts/check_paper_drift.py` flags any number that does not trace
to the baseline or to those files.

Strength levels:
- **ESTABLISHED**: state plainly, no hedging beyond the given scope.
- **SUPPORTED**: state as a result, with its stated scope and caveat.
- **EXPLORATORY**: may appear only labelled as exploratory or descriptive, never in the abstract
  as a finding.
- **NULL**: report as a null or failed prediction; do not spin it as positive.
- **NOT CLAIMED**: must not be asserted anywhere, in text, captions or figure annotations.

## Main claims

| # | Claim | Strength | Evidence | Required scope or caveat |
|---|---|---|---|---|
| C1 | The query word's own tokens, in a sentence that asks nothing, encode `first_letter`, `last_letter` and `most_frequent_letter` above the word-only floor in all 16 models | ESTABLISHED | Table `tab_store`; 16/16 at both shot counts, both reads | Linear decodability, held-out words |
| C2 | Most of that word-level signal is already in the token embedding (layer 0): 73% (`last_letter`, last token, 0 shots), 82% (`most_frequent_letter`, span mean); 53–56% for `first_letter` | ESTABLISHED | `numbers.json` embedding_share | Consistent with prior work; frame as confirmation, not discovery |
| C3 | The word-level store does not grow reliably with scale | SUPPORTED | r = 0.47 (p = 0.064), −0.05, −0.16 | "not reliably": `last_letter` leans positive without significance |
| C4 | The question − no-question difference at the answer position grows with log parameters for all four prompt-sensitive tasks (`last_letter`, `most_frequent_letter`, `extraction`, `letter_before`), at 0 and 3 shots | ESTABLISHED | `tab_interaction`; Holm-adjusted p ≤ 0.016 (max 0.0153); within-family slopes and family-resampling CIs agree | Cross-model association over 16 related checkpoints, not a scaling law |
| C5 | That growth is confined to late layers (≥ 0.5 depth); early layers (≤ 0.25) show no scaling | ESTABLISHED | `tab_interaction` late/early columns; early p ≥ 0.43 | |
| C6 | An early bump (~10% depth) in the question arm is similar at every scale | SUPPORTED | `last_letter` early level 0.75, slope −0.002, p = 0.96 | Its source (copying spelling to the adjacent token) is a **plausible interpretation**, not shown |
| C7 | Distance between word and answer token cannot explain the scaling | SUPPORTED | Follows from C5 | |
| C8 | Small models' answer positions fall below their word-level store; large models' exceed it; (answer − store) grows with scale for `last_letter` and `most_frequent_letter` | SUPPORTED | p = 0.017 / 0.0007 (`last_letter`), 0.012 / 0.0016 (`most_frequent_letter`); Qwen3-0.6B +0.257 vs +0.307, phi-4 +0.434 vs +0.344 | Say "more **linearly** decodable"; the cause (computation, cross-token integration, re-encoding) is **not identified** |
| C9 | Against the word-only floor, `last_letter` retains ≈0 of its signal without the question at 3 shots (−0.009), versus 0.503 against the majority baseline | ESTABLISHED | `tab_retained` | |
| C10 | Property specificity: ~70% of the question effect is specific to the asked property for `last_letter` and `extraction`; ~40% for `most_frequent_letter`; 62% for `letter_before` | SUPPORTED | `tab_crosstask` | The design does not separate a general character routine from property-specific routing |
| C11 | Original count tasks fail the floor; length matching alone is insufficient; removing repetition and query-letter shortcuts makes `counting_lm` measurable (14/16 at 3 shots) | ESTABLISHED | `tab_nuisance`, §7 | |
| C12 | Rebuilt count tasks are only partly question-driven and show no significant question × scale interaction | SUPPORTED | p = 0.25 / 0.12, 0.97 / 0.06 | |
| C13 | Representation–readout gaps show no single scaling trend; `last_letter` narrowing is a ceiling effect | SUPPORTED | `tab_gap`; logit r = −0.43, p = 0.09 (pre-registered criterion failed) | Other per-task trends are uncorrected and individually not claims |
| C14 | Answer-token span beats a random subspace (+0.12) but barely beats PCA (+0.023) | SUPPORTED | §8 | Projection deficit vs gap correlation does **not** show the span causes the gap |
| C15 | Post-trained models' span beats PCA by more and loses less under projection at 3 shots | EXPLORATORY | p = 0.012 / 0.003; not at 0 shots | Confounded with family; unplanned |
| C16 | Crossing depth becomes shallower with scale (`most_frequent_letter`, 3 shots) | EXPLORATORY | r = −0.76, p = 0.0015; other cells n.s. | Not a finding |
| C17 | A question placed before the word does not change the word's encoding beyond a prefix change | NULL (pre-registered prediction failed) | `tab_f3` | Prefix sensitivity up to ~0.05 |
| C18 | Estimator choice (held-out layer selection, cold start) does not change the earlier results | ESTABLISHED | optimism 0.001–0.019; v4 − v3 within ±0.021 | |

## Not claimed (must not appear)

- That the model **uses** the store, the late signal or the answer-token span (no causal intervention).
- That the late signal is **computation**, or that it is **routing** of the store.
- That larger models carry more **information** about the property (only: more linearly decodable).
- A scaling **law**, or population-level conclusions about model families.
- That gaps close for retrieval tasks and persist for computational ones.
- Anything about other languages, scripts, tokenizers, chain-of-thought or free generation.
- That the token-split or MLP-probe analyses were run (they were not; they are Limitations).

## Things that must stay

- All Limitations content (it may be reorganized, not weakened).
- Pre-registered criteria and their outcomes, including the failed ones (Appendix B).
- The statement that the word-level store confirms prior work.
- Holm correction where four-task claims are made; "uncorrected" labels on descriptive analyses.
