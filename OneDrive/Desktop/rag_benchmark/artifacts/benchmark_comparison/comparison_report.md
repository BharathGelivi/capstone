# X-RAG vs. RAGAS / RAGChecker / ARES -- Benchmark Comparison Report

This report compares the X-RAG Diagnostic Framework against three established
RAG evaluation baselines (RAGAS, RAGChecker, ARES's `ues_idp` lightweight
LLM-judge mode) over a 10-example labeled evaluation set
(`eval/eval_dataset.csv`), covering healthy pipeline runs and five injected
failure modes (`MISSING_CORPUS`, `RETRIEVAL_MISS`, `CHUNK_BOUNDARY`,
`UNSUPPORTED_GENERATION`, `GROUNDING_FAILURE`).

## Methodology Notes

- All three baselines' LLM-judge calls are routed through Hugging Face's
  OpenAI-compatible Inference Providers router, reusing this project's own
  `HF_TOKEN` -- no separate OpenAI/Bedrock key was used, keeping the
  comparison free to reproduce.
- ARES was run in **`ues_idp` mode** (a few-shot LLM-judge, no classifier
  training) rather than **`ppi` mode** (fine-tuned classifiers + a larger
  human-labeled gold set for statistically-corrected confidence intervals).
  This is a deliberate scope decision: `ppi` mode's training/annotation
  requirements are disproportionate to a 10-example evaluation set.
- `ragchecker` and `ares-ai` each required their own separate Python 3.10
  virtual environment, isolated both from this project's main Python 3.13
  venv and from each other (their pinned dependencies conflict). See
  `docs/RESEARCH_LOG.md` for the full environment investigation -- no Rust or
  C/C++ compiler installation was ultimately required for either.

## 1. Summary

| Framework | What it measures | Mean faithfulness-style score (n=10) | Approx. LLM calls / example |
|---|---|---|---|
| X-RAG | This project's own diagnostic pipeline: claim-level NLI verification against retrieved context, with root-cause attribution to a *specific pipeline stage* (retrieval / chunking / generation / grounding). | 0.8699 | ~1 local NLI model call per claim-sentence pair (no LLM judge at all) |
| RAGAS | LLM-as-judge, reference-free by default. Reports aggregate scores per metric (faithfulness, answer relevancy, context precision/recall, answer correctness) with no stage attribution. | nan | ~2-3 LLM judge calls per example (claim decomposition + per-claim/context judging) |
| RAGChecker | Claim-level precision/recall/F1 against a gold answer, plus a retriever-vs-generator metric split (claim_recall/context_precision vs. faithfulness/hallucination) -- attributes to *retriever or generator*, not further. | 0.8175 | ~5+ LLM judge calls per example (claim extraction x2 + 4 claim-checking passes, each internally self-consistency-checked) |
| ARES (ues_idp) | Lightweight LLM-judge: per-retrieved-document context relevance, plus answer relevance and answer faithfulness -- no stage attribution beyond per-document context judgments. | 0.2054 | ~1 LLM judge call per retrieved document for context relevance, +2 more per document if relevant |

## 2. Correlation

_No correlation data available (run scripts/analyze_agreement.py first)._

## 3. Agreement (Failure / No-Failure Classification)

_No agreement data available (run scripts/analyze_agreement.py first)._

## 4. Where X-RAG Adds Value Beyond Aggregate Scores

X-RAG's differentiating claim is **localization**: where RAGAS and ARES report aggregate scores per metric with no attribution to *where* in the pipeline a failure occurred, and RAGChecker attributes only as far as retriever-vs-generator, X-RAG's root cause analysis names a specific pipeline stage (`MISSING_CORPUS`, `RETRIEVAL_MISS`, `CHUNK_BOUNDARY`, `UNSUPPORTED_GENERATION`, or `GROUNDING_FAILURE`) with a reasoning chain justifying it.

No disagreement examples were captured to illustrate this (run scripts/analyze_agreement.py).

## 5. Threats to Validity: Failures Encountered

**ragas** -- 14 failure(s):
  - eval_id `1`: ragas exploded
  - eval_id `1`: ragas exploded
  - eval_id `1`: ragas exploded
  - eval_id `1`: ragas exploded
  - eval_id `1`: ragas exploded
  - eval_id `1`: ragas exploded
  - eval_id `1`: ragas exploded
  - eval_id `1`: ragas exploded
  - eval_id `1`: ragas exploded
  - eval_id `1`: ragas exploded

## 6. Diagnostic Accuracy (Root Cause Attribution)

Sections 1-5 above ask whether X-RAG's *scores* track the baselines'. This
section asks the more important question for a diagnostic tool: is X-RAG's
*diagnosis* actually correct? Compares `xrag_primary_cause` against
`eval/eval_dataset.csv`'s own `expected_failure_type` label for each example
(healthy examples have an implicit expected label of `UNKNOWN`, matching
X-RAG's own "no failure detected" vocabulary).

Evaluated on 10 of 40 labeled examples (a small pilot slice -- treat directionally, not as a powered result).

**Overall diagnostic accuracy: 0.700** (7/10 correct)

### Confusion Matrix (expected -> predicted)

| Expected \ Predicted | CHUNK_BOUNDARY | UNKNOWN |
|---|---|---|
| **RETRIEVAL_MISS** | 1 | 1 |
| **UNKNOWN** | 1 | 7 |

### Per-Category Precision / Recall / F1

| Category | Precision | Recall | F1 | Support (expected count) |
|---|---|---|---|---|
| CHUNK_BOUNDARY | 0.000 | N/A | N/A | 0 |
| RETRIEVAL_MISS | N/A | 0.000 | N/A | 2 |
| UNKNOWN | 0.875 | 0.875 | 0.875 | 8 |

### Mismatches
- eval_id `5`: expected `UNKNOWN`, X-RAG predicted `CHUNK_BOUNDARY`
- eval_id `9`: expected `RETRIEVAL_MISS`, X-RAG predicted `CHUNK_BOUNDARY`
- eval_id `10`: expected `RETRIEVAL_MISS`, X-RAG predicted `UNKNOWN`

## 7. Aggregation Strategy Ablation

Tests whether the choice of evidence-aggregation strategy (`top1` /
`max_pool_top3` / `concat_top3`) materially affects diagnostic accuracy.
All three strategies are evaluated in a single pass (the expensive NLI
scoring is done once; only the combination rule varies). See
`scripts/ablate_aggregation_strategy.py` for full methodology.

_No ablation data available (run scripts/ablate_aggregation_strategy.py first)._

## 8. Confidence Calibration

Checks whether `RootCauseAnalysis.diagnosis_confidence` is well-calibrated:
when X-RAG reports confidence=0.93, is it actually correct ~93% of the time?
Bins examples by confidence and reports empirical accuracy per bin plus
Expected Calibration Error (ECE). See
`scripts/evaluate_confidence_calibration.py`.

Evaluated on 10 examples. **Expected Calibration Error (ECE): 0.2240** (lower = better; 0 = perfectly calibrated).

A well-calibrated diagnostic tool would show `empirical_accuracy ≈ mean_confidence` in every bin. A large gap signals over- or under-confidence in the diagnosis.

| Confidence Bin | N | Mean Confidence | Empirical Accuracy | Gap |
|---|---|---|---|---|
| [0.00, 0.50) | 0 | — | — | — |
| [0.50, 0.70) | 0 | — | — | — |
| [0.70, 0.85) | 0 | — | — | — |
| [0.85, 0.95) | 10 | 0.924 | 0.700 | 0.224 |
| [0.95, 1.01) | 0 | — | — | — |

> **Interpretation note:** all 9 examples fell in the `[0.85, 0.95)` bin because the `RootCauseReasoner` emits a fixed confidence of 0.90 or 0.93 depending on whether evidence is ambiguous -- not a posterior probability learned from data. True per-example calibration requires a larger, more diverse eval set and a confidence model that varies more continuously.

## 9. Reasoning Chain Validation

Diagnostic accuracy (#6) checks whether the final *label* is correct. This
section checks whether the *reasoning* is self-consistent (automated checks)
and provides the human-rating worksheet for independent validation. See
`scripts/evaluate_reasoning_consistency.py`.

**Automated self-consistency rate: 1.000** (10/10 examples pass all automated checks: correct stage → FAIL status mapping, primary cause named in chain, all FAIL stages mentioned).

> ⚠️ **This is an AUTOMATED self-consistency check (does the narrative match its own underlying data), NOT a substitute for human judgment on whether the reasoning is actually sound. See reasoning_chain_worksheet.csv for the human-rating step, which has not yet been completed.**

### Per-Example Automated Consistency Checks

| Eval ID | Consistent | primary_cause_stage_is_fail | chain_names_primary_cause | all_fail_stages_named |
|---|---|---|---|---|
| 1 | ✓ | N/A | N/A | ✓ |
| 2 | ✓ | N/A | N/A | ✓ |
| 3 | ✓ | N/A | N/A | ✓ |
| 4 | ✓ | N/A | N/A | ✓ |
| 5 | ✓ | ✓ | ✓ | ✓ |
| 6 | ✓ | N/A | N/A | ✓ |
| 7 | ✓ | N/A | N/A | ✓ |
| 8 | ✓ | N/A | N/A | ✓ |
| 9 | ✓ | ✓ | ✓ | ✓ |
| 10 | ✓ | N/A | N/A | ✓ |

**Human validation status:** The `reasoning_chain_worksheet.csv` has been generated at `artifacts/benchmark_comparison/reasoning_chain_worksheet.csv` with one row per trace, including the full reasoning chain and empty `sound_yn` / `notes` columns. This worksheet has **not yet been filled in** -- human validation remains a pending step.

## 10. Cost / Latency Tradeoff

Turns the previously-qualitative cost comparison into real measured numbers.
X-RAG's per-stage latency has always been captured; baseline wall-clock
latency requires the instrumented `run_baseline_comparison.py` columns
(`ragas_latency_ms` etc.). See `scripts/evaluate_cost_latency.py`.

Measured over 10 X-RAG traces (all real wall-clock times, not approximated call counts).

### X-RAG Per-Stage Latency (mean across examples)

| Stage | Mean Latency |
|---|---|
| Retrieval | 3,956 ms |
| Generation | 2,296 ms |
| Verification (NLI) | 810,334 ms |
| **Total** | **816,587 ms** |

> **Note:** The verification stage dominates because the local NLI model scores every claim × sentence pair; retrieval and generation are network-bound calls that are comparatively fast.

### Baseline Latency (where instrumented)

| Baseline | N instrumented | Mean Latency |
|---|---|---|
| RAGAS | 1 (of 10 total) | 143,248 ms |
| RAGCHECKER | 0 (of 10 total) | N/A |
| ARES | 0 (of 10 total) | N/A |

> **Baseline latency note:** Examples run before the latency instrumentation columns (`ragas_latency_ms`, `ragchecker_latency_ms`, `ares_latency_ms`) were added to the results schema show 0 instrumented examples. Re-running those examples with the updated `run_baseline_comparison.py` will populate real wall-clock data.

### LLM Call Approximation (static estimate, not measured)

| Framework | Approx. LLM calls/example |
|---|---|
| X-RAG | ~1 local NLI model call per claim-sentence pair (no LLM judge at all) |
| RAGAS | ~2-3 LLM judge calls per example (claim decomposition + per-claim/context judging) |
| RAGChecker | ~5+ LLM judge calls per example (claim extraction x2 + 4 claim-checking passes, each internally self-consistency-checked) |
| ARES (ues_idp) | ~1 LLM judge call per retrieved document for context relevance, +2 more per document if relevant |
