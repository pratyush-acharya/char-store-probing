# Citation audit

## v4 update (2026-10-09)

The LaTeX source cites 37 keys; every key resolves to one bibliography entry and every entry is
cited (checked by comparing `\cite` keys in `main.tex` and `sections/*.tex` with `references.bib`).

Added in v4, each checked against its primary record:

- Kaushal and Mahowald, *What do tokens know about their characters and how do they know it?*,
  NAACL 2022, pp. 2487–2507, doi:10.18653/v1/2022.naacl-main.179:
  [ACL Anthology](https://aclanthology.org/2022.naacl-main.179/). Cited for: token embeddings encode
  spelling, improving with model size.
- Itzhak and Levy, *Models In a Spelling Bee*, NAACL 2022, pp. 5061–5068,
  doi:10.18653/v1/2022.naacl-main.373: [ACL Anthology](https://aclanthology.org/2022.naacl-main.373/).
  Cited for: RoBERTa and GPT-2 embedding layers suffice to spell up to a third of the vocabulary.
- Sato and Sasano, *How Do Language Models Acquire Character-Level Information?*, EACL 2026,
  pp. 5987–5997, doi:10.18653/v1/2026.eacl-long.282:
  [ACL Anthology](https://aclanthology.org/2026.eacl-long.282/). Cited only for its stated factors
  (tokenization-related and tokenization-independent); its abstract makes no claim about where
  character information lives, and v4 attributes none to it.
- Holm, *A Simple Sequentially Rejective Multiple Test Procedure*, Scandinavian Journal of
  Statistics 6(2):65–70, 1979. Cited for the correction procedure.

Removed in v4: `manifold_counting` (no longer discussed).

Newly cited from the existing bibliography: `wei2022cot`, `kojima2022zeroshot` (Limitations).

## v3 record (2026-08-17)

Checked on 2026-08-17 against primary venue, author, or arXiv records. The LaTeX source contains
34 citation keys; every key resolves to one bibliography entry and every bibliography entry is cited.

## Corrections made in v3

- KAPPA is listed as an ICML 2026 paper rather than a 2025 proceedings paper:
  [arXiv:2509.23782](https://arxiv.org/abs/2509.23782).
- *The Strawberry Problem* now uses its EMNLP 2025 pages and DOI:
  [ACL Anthology](https://aclanthology.org/2025.emnlp-main.1434/).
- Hiraoka and Inui now uses its Findings of EMNLP 2025 pages and DOI:
  [ACL Anthology](https://aclanthology.org/2025.findings-emnlp.719/).
- CharBench is listed under its AAAI 2026 publication year rather than 2025:
  [arXiv:2508.02591](https://arxiv.org/abs/2508.02591).
- The Tuned Lens author order was corrected to the current primary record:
  [arXiv:2303.08112](https://arxiv.org/abs/2303.08112).
- The Dhar and Bisazza poster title and work type were corrected.
- The Brown Corpus source and primary reports for Qwen3, Llama 3, Gemma 2, Phi-4, OLMo 2,
  OLMoE, and Olmo Hybrid were added because the dataset and model families should be cited explicitly.

## Claim-scope checks

- The early-encoding paper explicitly discusses its decision not to use control tasks; v3 describes
  that methodological position without endorsing it:
  [arXiv:2604.00778](https://arxiv.org/abs/2604.00778).
- *When Models Manipulate Manifolds* studies a fixed-width line-breaking/counting setting. V3 no
  longer presents it as direct validation of repeated-letter counting in natural words:
  [Transformer Circuits](https://transformer-circuits.pub/2025/linebreaks/index.html).
- The repeated-token paper's late-MLP and high-probe-accuracy results are described only in its
  repeated-token setting:
  [arXiv:2605.09239](https://arxiv.org/abs/2605.09239).
- Garcia's result concerns individual digit-token directions, whereas this paper tests their joint
  span; v3 states these as different geometric measurements:
  [arXiv:2605.03258](https://arxiv.org/abs/2605.03258).

## Model-report records

- [Qwen3](https://arxiv.org/abs/2505.09388)
- [Llama 3](https://arxiv.org/abs/2407.21783)
- [Gemma 2](https://arxiv.org/abs/2408.00118)
- [Phi-4](https://arxiv.org/abs/2412.08905)
- [OLMo 2](https://arxiv.org/abs/2501.00656)
- [OLMoE](https://arxiv.org/abs/2409.02060)
- [Olmo Hybrid](https://arxiv.org/abs/2604.03444)

Publication metadata can change for work that remains an arXiv preprint. Recheck those entries at
submission time.
