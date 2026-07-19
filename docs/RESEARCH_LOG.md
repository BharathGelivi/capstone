# RAG Benchmark - Research & Development Log

## Session: July 9, 2026 - RAG Core Enhancements & Keyword Fixing

### What We Did
- **PDF Extraction Fixes:** Replaced `llama-index`'s default PDFReader with `PyMuPDFReader` (via `pymupdf`). We encountered issues where text blocks from legal documents (BNS, BNSS) were either poorly formatted or missing entirely due to simple extraction boundaries. `PyMuPDFReader` drastically improved extraction quality.
- **Hybrid Retrieval Implementation:** The dense retriever was missing exact keyword matches (e.g., retrieving exact section numbers in legal text). We implemented a **Hybrid Retriever** using BM25 Sparse Search (`rank_bm25`) combined with the existing ChromaDB Dense Search, fused using Reciprocal Rank Fusion (RRF).
- **BM25 Tokenization Fixes:** 
  - **Punctuation Bug:** The default `.split()` tokenizer was failing because `399.` was tokenized differently than `399`. We fixed this by using a regex `r'\w+'` tokenizer.
  - **Stopword Skew:** BM25 over-weighted common words (like *in, about, what, does, section*). The exact section number `399` was buried by documents that repeatedly used the word `in`. We fixed this by actively filtering standard English stopwords inside the BM25 tokenizer.
- **LLM Hallucination Fixes:** When queried about "section 399", the `Llama-3-8B-Instruct` model returned hallucinatory/confused outputs ("there is no mention...").
  - **Prompt Relaxation:** Adjusted the system prompt to explicitly explain that legal texts use bare numbers (e.g. `399. (1)`) to refer to sections.
  - **Chat Templating:** Shifted `HuggingFaceInferenceAPI` from using `.complete(prompt_string)` to `.chat(messages_array)`. This forced Llama-3 to use its native instruction-tuning chat template (`<|start_header_id|>...`), permanently resolving the hallucination.
- **Cross-Encoder Reranking:** BM25+Dense retrieval successfully retrieves a broad recall set (Top-20). We added a `BAAI/bge-reranker-base` Cross-Encoder to rerank these 20 candidates by directly scoring the query against the chunk text, narrowing down to the final Top-5 chunks (Precision).
- **Metadata Fixes:** Extracted exact page numbers from PDF documents and properly attached them to `RetrievalResult` objects to make the RAG pipeline easily verifiable by humans.

### Problems Faced
1. **Extraction:** PDF readers failing on complex Indian Legal formats.
2. **Dense Search Blindspots:** Semantic embeddings fail hard at exact keyword matching (especially alphanumeric codes/sections).
3. **BM25 Skew:** Raw BM25 breaks down when querying with natural language sentences full of filler words, assigning extreme TF scores to irrelevant chunks.
4. **LLM formatting:** Using base-completion APIs on Instruct-tuned models completely ruins their ability to follow instructions.

### What We Plan To Do Next
- **Diagnostic Framework Integration:** Start building the diagnostic builder (Claim Engine) that maps the `RAGTrace` against the `ChunkRegistry` to automatically flag hallucination, context drift, and retrieval drop-offs.
- **Eval Framework:** Introduce known failure conditions to the chunks and test if the diagnostic tool can catch them automatically using the traces.

---

## Session: RAGTrace Builder

### Module
`src/rag_trace.py` and `src/test_rag_trace.py`

### Design Decisions
- **Strictly Factual:** `RAGTrace` is designed to be a standardized execution record containing only facts (e.g., timestamps, chunk IDs, similarity scores, full prompts, generation outputs) without performing any active evaluation. 
- **Omitted Chunk Text:** We explicitly chose NOT to duplicate full chunk text within the trace object. Instead, we store lightweight references (`chunk_id`, rank, similarity). The canonical chunk text always lives in the `ChunkRegistry`, avoiding massive trace files and ensuring a single source of truth.
- **Date Partitioning:** Traces are serialized to JSON and stored in `artifacts/rag_traces/YYYY-MM-DD/` to keep executions organized over time.

### Trade-offs
- **Coupling vs. Storage:** By referencing chunks instead of embedding them, analyzing a trace now requires access to the `ChunkRegistry`. This slightly increases the complexity of downstream diagnostic tools but dramatically reduces the storage footprint of each trace.

### Future Improvements
- Add placeholder `diagnostics` block to store evaluation results once the Claim Engine is implemented.
- Implement a Trace Viewer UI or CLI to easily read and navigate through historical RAG traces.

### How RAGTrace Enables Claim-Level Diagnostics
By capturing the exact state of the pipeline (including configuration, time metrics, exactly which chunks were retrieved, and the exact prompt passed to the LLM), `RAGTrace` acts as a perfect "black box flight recorder". Future modules (like a Claim Engine) will consume this exact artifact, fetch the referenced chunks from the registry, and systematically evaluate the `generated_answer` against the provided context, completely detached from the execution environment.

---

## Session: Claim Representation Framework (Phase 2.1)

### Module
`src/claims.py` and `src/test_claims.py`

### Implementation
- Defined `ClaimType` and `VerificationComplexity` Enums.
- Created the `Claim` and `ClaimSet` dataclasses.
- Implemented `ClaimFactory` for deterministic `claim_id` generation (`<trace_id>_C001`).

### Design Decisions
- **Strict Separation of Representation vs. Evaluation:** The `Claim` object explicitly *lacks* fields like `confidence_score`, `support_label`, or `hallucination_status`. The purpose of this module is solely to represent the facts of an extracted claim, leaving evaluation entirely to Phase 2.2.
- **Verification Complexity (VCL) as a Property:** `VCL` is tied to the claim itself because complexity is an inherent linguistic property of the claim (e.g. how hard it is to verify), rather than the result of the verification process.
- **Hashing (`claim_hash`):** Included to allow deduplication of functionally identical claims across different generations or traces, aiding in aggregative analytics.
- **Extensive Optionality:** Since claims are often constructed incrementally in a pipeline, many fields (`normalized_text`, `claim_type`, `claim_hash`) are made optional so they can be hydrated as the claim moves through the system.

---

## Session: Claim Decomposer (Phase 2.1 Module 2)

### Module
`src/claim_decomposer.py` and `src/test_claim_decomposer.py`

### Implementation
- Added `CandidateClaim` and `CandidateClaimSet` dataclasses.
- Implemented `ClaimDecomposer` using Gemini 1.5 Pro to extract claims.
- Integrated standard LLM prompting techniques specifying a formal definition of atomic claims and 8 distinct strict rules.
- Supported extracting debug reasoning dynamically via `CLAIM_DECOMPOSER_DEBUG`.

### Design Decisions
- **Separation from Validated Claims:** We explicitly created `CandidateClaim` and separated it from the standard `Claim` object in Phase 2.1. Candidate claims represent the LLM's raw initial decomposition of a text. This intermediate stage prevents malformed or unverified outputs from contaminating the canonical claim registry before normalization and validation phases operate.
- **Strict JSON Enforcement:** The LLM was prompted heavily to only return a strict JSON array. By forcing a schema (`claim_text`, `source_sentence`, `sentence_id`), downstream automated pipelines can securely parse and manipulate the generated factual items. We implemented a 1-retry fallback for decode errors.
- **Python-Side Indexing:** We decided to calculate `character_start` and `character_end` in Python via string-matching instead of asking the LLM to output them. LLMs frequently hallucinate exact string indices, which leads to misalignment when highlighting text in frontend tools.
- **Intentional Exclusion of Validation:** Verification, typing, and complex classification (e.g., complexity metrics) are explicitly avoided at this stage. The sole responsibility of the decomposer is *fracturing* the text into atomic blocks. Pushing validation here would create a monolithic prompt and increase latency, violating the single responsibility principle.

---

## Session: Assumption Checker (Phase 3.1)

### Module
`src/assumption_checker.py` and `src/test_assumption_checker.py`

### Implementation
- Added `PipelineStage` and `AssumptionStatus` Enums.
- Created `AssumptionResult` and `AssumptionMatrix` dataclasses to store evaluation outcomes.
- Implemented `AssumptionChecker` utilizing deterministic, rule-based logic to evaluate `RAGTrace`, `ClaimSet`, and `VerificationSummary` across 5 key pipeline stages (Corpus, Retriever, Chunking, Generator, Grounding).

### Design Decisions
- **Rule-Based Evaluation over LLMs:** We explicitly chose to implement deterministic rule-based checking rather than using an LLM. This ensures consistent, reproducible evaluation metrics and significantly reduces latency and cost.
- **Assumptions vs. Direct Failure Prediction:** We evaluate *pipeline assumptions* rather than directly predicting failures because failures in RAG are often cascading. By checking if foundational assumptions hold true (e.g., "The retriever found relevant evidence", "The generator only used retrieved evidence"), we can logically deduce where the pipeline broke. Direct failure prediction often conflates symptoms (like hallucinated text) with root causes (like poor retrieval forcing the model to guess).
- **Traceability:** The `AssumptionMatrix` acts as a discrete, serializable artifact mapping directly to a specific `trace_id`. This allows downstream modules to analyze pipeline health historically.

---

## Session: Refactoring Pipeline State Analyzer (Phase 3.1 Refined)

### Design Decisions
- **Separation of Observation from Reasoning:** The `AssumptionChecker` was refactored into the `PipelineStateAnalyzer` to strictly isolate factual observations from interpretative reasoning. Words like "hallucination" or "failure" have been completely removed from observations. The analyzer strictly reports "what happened" (e.g., "Three claims remain unsupported") rather than "why it happened" (e.g., "Generator Hallucinated"). This ensures downstream modules receive untainted facts.
- **Preference for UNKNOWN:** We explicitly prefer returning a status of `UNKNOWN` rather than engaging in speculative inference. If evidence is ambiguous, the factual layer must reflect that ambiguity, leaving deduction to the future Root Cause Reasoner.
- **Immutability of the Pipeline State Matrix:** The newly designated `PipelineStateMatrix` (PSM) is designed as a rigid, immutable research artifact (versioned, e.g., "1.0"). It represents a frozen snapshot of observable pipeline state immediately post-verification.
- **Read-Only Consumption:** By saving the PSM to disk (e.g., `TRACE_xxxxx.json`), we enforce a strict architectural boundary. Future diagnostic or reasoning modules must consume this serialized artifact rather than recomputing the observations themselves, guaranteeing consistency across experiments.

---

## Session: Root Cause Reasoner (Phase 3.2)

### Module
`src/root_cause_reasoner.py` and `src/test_root_cause_reasoner.py`

### Implementation
- Added `FailureType` Enum to represent standardized pipeline failures (`MISSING_CORPUS`, `RETRIEVAL_MISS`, `CHUNK_BOUNDARY`, `UNSUPPORTED_GENERATION`, `GROUNDING_FAILURE`, etc.).
- Created the `RootCauseAnalysis` dataclass to encapsulate the reasoning artifact.
- Implemented `RootCauseReasoner` to consume a read-only `PipelineStateMatrix` and evaluate causality sequentially.

### Design Decisions
- **Separation of Reasoning from Observation:** Causal reasoning is entirely detached from the observation layer (`PipelineStateAnalyzer`). The PSM strictly provides facts, and the Reasoner provides interpretations. This decoupling allows us to swap reasoning algorithms (e.g., deterministic vs. LLM-based) without altering the foundational facts.
- **Earliest Violated Assumption as Primary Cause:** Since RAG pipelines are sequential and highly coupled, a failure upstream almost guarantees a failure downstream (e.g., a retrieval miss forces the generator to hallucinate). We treat the *earliest* `FAIL` state in the pipeline execution order (Corpus -> Retriever -> Chunking -> Generator -> Grounding) as the primary root cause. Subsequent failures are cataloged as secondary effects.
- **UNKNOWN States are Skipped:** `UNKNOWN` states indicate missing or ambiguous evidence. We do not infer causality from them; they are skipped during traversal to avoid speculative, low-confidence conclusions.

---

## Session: Corrective Action Engine (Phase 3.3)

### Module
`src/corrective_action_engine.py` and `src/test_corrective_action_engine.py`

### Implementation
- Created the `ActionCategory` Enum to classify actions structurally.
- Developed the `CorrectiveAction` and `CorrectiveActionPlan` dataclasses.
- Implemented the `CorrectiveActionEngine` which consumes the `RootCauseAnalysis` artifact.

### Design Decisions
- **Separation of Correction from Diagnosis:** Corrective action generation is detached from Root Cause Analysis. This ensures the reasoning layer acts solely as a diagnosis engine, while the corrective layer maps those diagnoses to actionable engineering tasks. This decoupling aligns with our strict single-responsibility architecture across the framework.
- **Deterministic, Evidence-Backed Actions:** We chose to use a static, deterministic lookup table rather than an LLM for generating actions. Because RAG architectural failures (e.g., `RETRIEVAL_MISS`, `CHUNK_BOUNDARY`) map to well-known engineering solutions (e.g., "Increase Top-K", "Increase Overlap"), using an LLM introduces unnecessary latency, cost, and risk of hallucinated advice. Deterministic mapping guarantees consistent, proven engineering tradeoffs are presented to the developer, strictly backed by observed evidence.
- **Iterative Improvement through Success Metrics:** Each corrective action includes an expected improvement and a quantifiable success metric. This grounds the diagnostic framework in measurable software engineering practices, ensuring that changes to the RAG architecture can be empirically verified against future `RAGTrace` evaluations.

---

## Session: Diagnostic Report Data Model (Phase 4.1)

### Module
src/report.py and src/test_report.py`n
### Implementation
- Created the DiagnosticEvaluationReport canonical data model and nested dataclasses (FrameworkMetadata, ExecutiveSummary, PipelineStageResult, EvaluationMetrics, etc.).
- Implemented strict serialization logic for the final report artifact.

### Design Decisions
- **Separation of Schema from Generation:** The report schema is strictly defined as an immutable data structure. The generation logic (which will calculate metrics and merge artifacts) is explicitly pushed to a future Report Builder phase. This enforces the Single Responsibility Principle.
- **Immutable Artifacts for Reproducibility:** By forcing the final report to be serialized and saved as an immutable JSON artifact (TRACE_xxxxx.json), we guarantee absolute reproducibility. Researchers can load and inspect any historical evaluation exactly as it was generated.
- **Separation of Rendering from the Object:** Rendering logic (Markdown, HTML, PDF) is intentionally kept out of the report object. The report is pure structured data. This allows front-ends or CI/CD pipelines to consume the raw data and render it dynamically without being coupled to hardcoded text formats.

---

## Session: Diagnostic Report Presenter (Phase 4.3)

### Module
src/report_presenter.py and src/test_report_presenter.py`n
### Implementation
- Created the DiagnosticReportPresenter class to render the diagnostic evaluation report into Console, Markdown, and HTML.
- Extracted evidence traceability paths using the evidence_analysis records and root cause conclusions.
- Added an Artifacts Generated Appendix to strictly track file paths for RAG traces, pipeline state matrices, and reports.

### Design Decisions
- **Separation of Presentation from Reasoning:** Presentation logic must never perform causal inference or metrics calculation. By separating presentation into the final layer, any front-end UI can reliably consume the same DiagnosticEvaluationReport object without implementing custom diagnostic logic.
- **Traceability for Explainability:** The Evidence Traceability section is critical to establishing trust. It explicitly maps every system failure back to the specific claim, text chunk, and pipeline stage that caused it, preserving total provenance.
- **Format Independence:** By generating native Markdown and semantic HTML directly from the deterministic report object, the framework supports headless execution, CI/CD pipeline integration, and automated research paper assembly.

---

## Session: API Layer + Repository Refactoring (Phase 4.5)

### Implementation
- Refactored repository structure to isolate 	ests/, configs/, rtifacts/, data/, and docs/.
- Migrated all unit tests out of the main src/ logic.
- Split src/config.py into distinct modular pieces: configs/models.py, configs/pipeline.py, configs/thresholds.py, and configs/api.py.
- Implemented src/runner.py to seamlessly orchestrate the pipeline.
- Integrated a comprehensive FastAPI layer in src/api.py and un_api.py.
- Standardized internal logging via src/logger.py.

### Design Decisions
- **API-First Architecture:** Shifting to an API-first approach drastically increases the flexibility of the diagnostic framework. It allows front-end web dashboards, CI/CD runners, and external agents to trigger analyses via JSON requests without integrating directly with the python codebase.
- **FastAPI over Flask:** FastAPI was chosen due to its native support for Pydantic (which pairs seamlessly with our existing dataclass architecture), automatic Swagger UI generation for effortless faculty demonstration, and high performance.
- **Repository Cleanup:** Isolating unit tests ensures the main src/ directory only contains deployment-ready framework code, minimizing bundle sizes for future distributions and streamlining developer onboarding.
- **Test Isolation:** Moving tests to a dedicated 	ests/ directory prevents production code from accidentally importing or relying on testing logic, enforcing strict separation of concerns.

---

## Session: July 18, 2026 - Generation Truncation Bug + Model Swap

### The Bug
Claim decomposition was silently extracting 0 claims from perfectly good, detailed
LLM answers, and answers themselves were occasionally cut off mid-word. Root cause:
`src/generator.py` and `src/claim_decomposer.py` both constructed
`HuggingFaceInferenceAPI(..., max_new_tokens=...)`, but that class (from
`llama-index-llms-huggingface-api`) has **no `max_new_tokens` field at all** --
the real field is `num_output` (`DEFAULT_NUM_OUTPUTS = 256` in llama-index-core).
Passing an unrecognized kwarg to this pydantic-based class is silently dropped,
so every generation call -- both the user-facing answer and the claim-JSON
response -- was capped at the library's 256-token default regardless of what
`configs/models.py`/`configs/pipeline.py` specified. Long answers were truncated
mid-sentence; the claim-decomposer's JSON array was truncated mid-object, which
the existing bracket-balancing JSON recovery couldn't fix (a dangling trailing
comma after a half-written object isn't recoverable by just appending closing
brackets).

### Fixes
- Changed `max_new_tokens=` to `num_output=` in both `Generator.__init__` and
  `ClaimDecomposer.__init__`. Added regression tests asserting the correct kwarg
  name so this can't silently reappear (`tests/test_generator.py`,
  `tests/test_claim_decomposer.py`).
- Hardened `ClaimDecomposer._robust_json_parse()` with a 5th recovery step:
  when a response is truncated mid-object, drop back to the last fully-closed
  top-level object (rightmost `}`), strip a dangling trailing comma, and close
  the array there -- recovering the N-1 complete claims instead of returning 0.
- Also fixed `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`'s NLI pipeline call:
  `return_all_scores=True` is deprecated in `transformers>=5` and silently
  returns only the top-1 label instead of all label scores; changed to
  `top_k=None`. Separately noted (not yet fixed): this specific checkpoint only
  has 2 labels (`entailment`/`not_entailment`), so `contradiction_score` and
  `neutral_score` have always silently defaulted to 0.0 -- `CONTRADICTED` can
  never actually be produced with this model regardless of code correctness.
- Swapped the default LLM from `mistralai/Mistral-7B-Instruct-v0.2` to
  `meta-llama/Meta-Llama-3-8B-Instruct`, which turned out to be unavailable via
  any Hugging Face Inference Provider on this account (`model_not_supported`).
  Settled on `Qwen/Qwen2.5-7B-Instruct` (confirmed working on free HF accounts,
  and already referenced for that reason in the pre-existing
  `tests/test_claim_decomposer.py` fixtures). Centralized this in
  `configs/models.py`'s `LLM_MODEL_NAME` and made `ClaimDecomposer` import it
  instead of hardcoding its own separate copy of the model name (previously
  the two had silently drifted apart).

### Known follow-up (not fixed, flagged for later)
Claim verification takes 2-7 minutes per claim in real runs: `find_best_evidence()`
in `src/claim_verifier.py` calls the NLI model one sentence at a time in a Python
loop, `VERIFICATION_BATCH_SIZE` is defined in config but never used. Benchmarked:
batching the same model gives only ~2x throughput on this CPU-only machine
(80s -> 40s for 10 pairs) -- the real cost is a large model with no GPU. A
smaller 3-way NLI model would fix both the speed and the 2-label ceiling above.

---

## Session: July 18, 2026 - Baseline Comparison (RAGAS / RAGChecker / ARES) Setup

### Goal
Produce an empirical X-RAG vs. RAGAS vs. RAGChecker vs. ARES comparison for the
paper. This session covered environment verification (don't trust memory of
library APIs -- they move fast) and the labeled eval dataset.

### Labeled eval dataset
Built `eval/eval_dataset.csv`: 40 question/gold-answer pairs authored directly
against verified text extracted from `data/bns.pdf`, `data/bnss.pdf`, `data/bsa.pdf`
(no invented legal content). Distribution: 25 direct/healthy questions, 4
`MISSING_CORPUS` probes (topics genuinely outside all three codes -- crypto
regulation, corporate tax, trademark law, company filings), 3 `RETRIEVAL_MISS`
probes (paraphrased away from the statute's own wording), 3 `CHUNK_BOUNDARY`
probes (short provisions sandwiched between longer neighbors, or explanatory
clauses spanning a page break -- candidates for being split awkwardly at a
512-character chunk size), 2 `UNSUPPORTED_GENERATION` probes (asking for a
specific number -- e.g. an exact fine amount -- that the source text
deliberately never specifies, tempting fabrication), and 3 `GROUNDING_FAILURE`
probes (false-premise questions asserting something that directly contradicts
the retrieved text, e.g. claiming murder's punishment is capped at ten years
when section 103 actually prescribes death or life imprisonment).

### Environment verification findings (installed packages, not memory)
- **ragas**: the PyPI "latest" (0.4.3) hard-imports
  `langchain_community.chat_models.vertexai.ChatVertexAI` at module load time.
  That submodule has been fully removed from `langchain-community` (migrated to
  the standalone `langchain-google-vertexai` package), so ragas>=0.4 fails to
  import at all as of 2026-07, independent of which langchain-community version
  pip resolves -- a genuine upstream packaging break, not a version-pinning
  problem we could have avoided. Pinned to `ragas==0.2.15`, which has the same
  unconditional import, so a local compatibility shim is required: a stub
  `ChatVertexAI` class dropped into the venv's own
  `langchain_community/chat_models/vertexai.py` (that file/submodule doesn't
  exist upstream anymore) so the import resolves. The stub is never functionally
  invoked -- Vertex AI is not used anywhere in this project. Shim content:
  ```python
  class ChatVertexAI:
      def __init__(self, *args, **kwargs):
          raise NotImplementedError(
              "ChatVertexAI is a compatibility stub only. "
              "Install langchain-google-vertexai for real Vertex AI support."
          )
  ```
  Confirmed the real ragas 0.2.15 schema directly from the installed metric
  objects (`_required_columns`), not from memory/docs: `SingleTurnSample` uses
  `user_input`/`response`/`retrieved_contexts`/`reference` -- not the older
  `question`/`answer`/`contexts`/`ground_truth` naming shown in some outdated
  tutorials. Also confirmed `ragas.llms.LangchainLLMWrapper` + a
  `langchain_openai.ChatOpenAI` pointed at `base_url="https://router.huggingface.co/v1"`
  lets ragas's judge LLM run through the project's existing `HF_TOKEN`, no
  OpenAI key needed; and `ragas.embeddings.LlamaIndexEmbeddingsWrapper` wraps
  our already-loaded llama-index embedding model directly, so no second
  embedding model needs to be loaded for ragas's embedding-based metrics.
- **ragchecker** and **ares-ai** (the actual current PyPI package for ARES --
  not a git install, contrary to older documentation; confirmed via the
  current repo README) both initially looked like they needed a real Rust and
  C/C++ compiler respectively, on this project's main Python 3.13 venv:
  `ragchecker -> refchecker -> litellm==1.92.0` has no prebuilt wheel for
  Python 3.13 (needs Rust to build from source), and `ares-ai` pins
  `numpy<2.0`, which has no prebuilt wheel for Python 3.13 on Windows (needs a
  C/C++ compiler to build from source).
- **Resolved without installing any compiler.** This machine already had
  Python 3.10 installed separately (`py -0p` showed it registered alongside
  3.13). Packages this old (numpy<2.0, scipy<1.11, torch<2.0, litellm-era
  pins) were built and published with wheels for that Python/OS combination;
  Python 3.13 support for them simply doesn't exist yet upstream. Two findings
  made this fully workable:
  - **ragchecker**: `pip install ragchecker` alone resolves `litellm` to
    `1.92.0` -- the actual current PyPI "latest" -- which turns out to have
    **no prebuilt wheel published for any platform** (an upstream release
    gap, not a Python-version problem). `litellm==1.91.3` (one version back)
    still satisfies refchecker's `litellm<2.0,>=1.49` constraint and does have
    a wheel; installing it explicitly before `ragchecker` makes pip accept the
    wheel-based version instead of trying to build the broken one from
    source. No Rust needed at all. See `requirements-eval-ragchecker.txt`.
  - **ares-ai**: installs cleanly on Python 3.10 with prebuilt wheels straight
    through (torch, scipy, numpy, transformers, spacy -- none needed
    compilation). Two remaining snags, both packaging quirks unrelated to
    compilers: (1) the install reliably fails on its *first* attempt with an
    `OSError` on a deeply-nested jupyterlab static asset path -- a Windows
    `MAX_PATH` limitation, not a real package problem; simply re-running the
    same install command a second time completes normally using the cached
    downloads. (2) `ares-ai`'s own code calls `datasets.load_metric`, removed
    from modern `datasets` (moved to the separate `evaluate` package);
    `datasets==2.19.0` is the last version that still has it. See
    `requirements-eval-ares.txt`.
  - **ragchecker and ares-ai must live in separate venvs from each other**,
    not just separate from the main project venv: installing both together
    creates real, unavoidable conflicts (ragchecker's transitive deps want
    newer `transformers`/`scikit-learn`/`openai`; ares-ai pins exact older
    versions of the same three packages). `venv_eval_ragchecker/` and
    `venv_eval_ares/` (both gitignored, both Python 3.10) keep them isolated.
- Decision: HF-hosted judge only (no OpenAI/Bedrock key) for all baselines that
  support it, per user preference, to keep this comparison free to run.

### Status (superseded -- see the session entry below for the final state)
`scripts/baseline_adapters/ragas_adapter.py` (+ `hf_judge.py` shared helper)
built and unit-tested (no network calls in tests), installed in the main
project venv. `ragchecker` and `ares-ai` both installed and import-verified in
their own dedicated Python 3.10 venvs (`venv_eval_ragchecker/`,
`venv_eval_ares/`); their adapters are the next piece of work.

## Session: July 18, 2026 - Benchmark Comparison Pipeline Complete (Runner, Agreement Analysis, Report Generator)

### What We Did
Completed the remaining pieces of the X-RAG vs. RAGAS/RAGChecker/ARES
comparison, building on the environment setup and adapters documented above:
- **`scripts/run_baseline_comparison.py`**: the resumable runner. For each row
  in `eval/eval_dataset.csv` it runs the question through X-RAG's own pipeline
  (retrieval -> generation -> claim verification -> root cause analysis),
  then computes RAGAS scores in-process (same venv), and RAGChecker/ARES
  scores via `subprocess` calls into their isolated Python 3.10 venvs (JSON
  file handoff in/out, since objects can't cross a venv boundary). A
  `manifest.json` tracks completed eval IDs so a re-run after an interruption
  or an HF 402 (credits exhausted) skips already-processed rows instead of
  redoing them. One baseline's failure is caught and logged
  (`failures.log`) without aborting the other baselines for that example.
  Supports `--dry-run` (build everything but skip actual model/LLM calls),
  `--limit`, `--skip-ragchecker`, `--skip-ares`.
- **`scripts/analyze_agreement.py`**: pure-Python (no numpy/scipy) Pearson
  correlation between X-RAG's `avg_entailment_score` and RAGAS/RAGChecker
  faithfulness-style scores; a binary failure/no-failure confusion matrix +
  Cohen's kappa between X-RAG's `primary_cause != UNKNOWN` verdict and each
  baseline's own threshold-based failure flag; and a `disagreements.csv`
  listing every example where X-RAG and a baseline disagree, each row
  enriched with X-RAG's actual reasoning chain (read back from the saved
  `RootCauseAnalysis` artifact) -- this is the qualitative material a paper's
  Discussion section would walk through.
- **`scripts/generate_comparison_report.py`**: renders everything above into
  one self-contained `comparison_report.md` -- summary table (what each
  framework measures, mean scores, approximate LLM-call cost per example),
  the correlation/agreement numbers, a "where X-RAG adds value beyond
  aggregate scores" section arguing the *localization* differentiator (stage
  attribution X-RAG has that RAGAS/ARES lack entirely and RAGChecker only
  partially has, retriever-vs-generator), backed by real disagreement
  examples, and an honest failures/threats-to-validity section parsed
  straight from `failures.log`. Every helper degrades gracefully to an
  explanatory placeholder ("run scripts/analyze_agreement.py first") rather
  than crashing when an upstream artifact is missing, since a full live run
  hadn't completed yet at the time this was built (see below).

### Problems Faced & Fixes
- A test in `test_run_baseline_comparison.py` mocked `subprocess.run` reading
  the output path from the wrong index of the constructed command list
  (`cmd[3]` instead of `cmd[4]`) -- caught before it could mask a real bug,
  fixed to match the actual `[python_exe, "-m", module, in_path, out_path]`
  shape.
- `compute_ragchecker_scores` initially assumed a nested
  `{"overall_metrics": {...}}` shape for RAGChecker's result object; reading
  `ragchecker/computation.py` and `ragchecker/metrics.py` directly (not
  bundled docs) showed `RAGResult.metrics` is actually a flat dict keyed
  directly by metric name -- fixed before it shipped.
- HF Inference Providers' free credits ran out mid-testing (HTTP 402) before
  a full live end-to-end run of `run_baseline_comparison.py` across all 40
  eval rows could complete. Per explicit user direction ("build the runner
  now, test later"), all three new scripts were instead validated with
  hand-constructed fixture data and mocked subprocess/API calls -- 189 tests
  pass across the whole suite, including 19 new tests for the report
  generator alone.

### Status
Coding phase of the benchmark comparison task is complete: environment setup,
all three adapters, the resumable runner, the agreement analysis, and the
report generator are built, documented, and unit-tested. **No full live run
has been executed yet** due to the HF credit exhaustion above -- running
`python -m scripts.run_baseline_comparison` (optionally `--dry-run` first),
then `analyze_agreement`, then `generate_comparison_report` end-to-end over
real API calls is the next step once credits are available.

## Session: July 19, 2026 - Switched Default LLM Provider to Groq (Free Tier)

### What We Did
A live dry-run confirmed HF Inference Providers' free credits were still at
zero (HTTP 402 on every call -- blocking X-RAG's own generation, not just the
baselines). Rather than wait for a credit purchase, added Groq's free-tier,
OpenAI-compatible API (`https://api.groq.com/openai/v1`) as an equally-real
zero-cost alternative, selectable via a new `configs/models.py` setting
(`LLM_PROVIDER = "groq"` vs `"huggingface"`) so switching back once HF
credits are purchased is a one-line config change, not a rewrite.

- **Model assignment spreads load across Groq's independent per-model rate
  limits** (30 RPM / ~1K RPD per model on the free tier) rather than routing
  everything through one model and risking contention:
  - `llama-3.3-70b-versatile` -- X-RAG's own answer generation
  - `llama-3.1-8b-instant` -- X-RAG's own claim decomposition (high daily
    quota, good for the higher call volume here)
  - `qwen/qwen3.6-27b` -- RAGAS's judge
  - `openai/gpt-oss-120b` -- RAGChecker's extractor/checker (its highest call
    volume of the three baselines)
  - `openai/gpt-oss-20b` -- ARES's judge (must contain the substring "gpt":
    ARES's own `ues_idp` code internally routes via a literal
    `"gpt" in model_choice` check, then sends that same string as the API
    `model=` parameter -- this exact model name happens to be hosted under
    the same name by both Groq and HF, so the ARES worker's existing
    HF-verified convention carries over unchanged)
- Wired the provider switch through `src/generator.py`, `src/claim_decomposer.py`
  (both branch on `LLM_PROVIDER`), a new `scripts/baseline_adapters/groq_judge.py`
  (mirrors the existing `hf_judge.py`), and `ragas_adapter.py`/
  `ragchecker_adapter.py`/`ares_worker.py` (branch on which API key is
  actually present -- `ragchecker_worker.py`/`ares_worker.py` run in isolated
  venvs without `configs/` installed, so they can't import `LLM_PROVIDER`
  directly).
- Generalized `src/env_check.py`'s interactive-prompt guards
  (`ensure_hf_token`/`ensure_hf_token_or_exit`) into provider-aware
  `ensure_llm_credentials`/`ensure_llm_credentials_or_exit`, so `query.py`/
  `run_pipeline.py`/`run_api.py`/`run_baseline_comparison.py` prompt for
  whichever key the active provider actually needs, instead of always
  demanding `HF_TOKEN`.
- **Removed the dead Gemini code path**: `claim_decomposer.py` had an
  unused `if model_name == "gemini"` branch (never selected by any config,
  no real wiring, leftover from an earlier exploration) and `.env`/`.env.example`
  carried an unused `GEMINI_API_KEY` placeholder. Uninstalled
  `llama-index-llms-gemini` and `google-generativeai` (the latter was already
  emitting a deprecation warning on every import) and dropped both from
  `requirements.txt`. `HF_TOKEN`/`hf_judge.py`/the whole `"huggingface"`
  provider path were deliberately kept, not removed -- purchasing HF credits
  and switching back remains a real, explicitly-planned next step.

### Problems Faced & Fixes
- Installing `llama-index-llms-groq` pulled in a resolver conflict:
  `transformers` got silently downgraded from 5.13.0 to 4.57.6 and
  `huggingface-hub` from 1.23.0 to 0.36.2, with pip warning
  `llama-index-llms-openai 0.6.26` was now incompatible with
  `llama-index 0.14.23`'s own constraint (`>=0.7.0`). This mattered because
  the NLI verifier's `top_k=None` fix (an earlier session) was specifically
  tied to a `transformers` version boundary -- re-verified live that
  `top_k=None` still constructs the zero-shot-classification pipeline
  correctly on 4.57.6, then ran the full test suite (still green) and
  smoke-imported every core pipeline module before trusting the install.
- All `@patch("src.generator.HuggingFaceInferenceAPI")` /
  `@patch("src.claim_decomposer.HuggingFaceInferenceAPI")` test patches broke
  silently in two ways once Groq became the default: the patch target no
  longer existed as a live module-level name in the intended sense (both
  classes are now imported unconditionally at module level specifically so
  patching stays stable regardless of which provider is active), and even if
  it had, the "groq" default provider meant the HF branch was never entered
  at all. Fixed by patching `Groq` for the provider-agnostic tests, and
  added a dedicated `@patch("...LLM_PROVIDER", "huggingface")` test alongside
  each Groq test so the HF fallback path (and its `num_output` vs.
  `max_new_tokens` regression fix) stays covered too.
- `ragchecker_worker.py`'s `--model` argparse default was hardcoded to
  `"Qwen/Qwen2.5-7B-Instruct"` (the old HF-only default), which meant it was
  always passed explicitly to `build_ragchecker()` and silently defeated that
  function's own provider-based default-model selection. Changed the
  argparse default to `None`.

### Status
All 191 tests pass (189 + 2 net-new from splitting the Groq/HF-fallback
`num_output` tests). Groq wiring is code-complete and unit-tested; a live
end-to-end run (`run_baseline_comparison.py` -> `analyze_agreement.py` ->
`generate_comparison_report.py`) is the next step once a `GROQ_API_KEY` is
added to `.env`.

## Session: July 19, 2026 - First Live Runs on Groq + Per-Example Detailed Diagnostic Report

### What We Did
- **First live runs against all 4 frameworks**, now that a real `GROQ_API_KEY`
  was added. Confirmed the whole pipeline (retrieval -> generation -> claim
  decomposition -> NLI verification -> RAGAS -> RAGChecker -> ARES) works
  end-to-end for real, across three examples.
- **Found and fixed a real judge-model bug live**: the two "reasoning" models
  originally picked for RAGAS/RAGChecker (`qwen/qwen3.6-27b`,
  `openai/gpt-oss-120b`) spend their token budget on an internal `<think>`
  block before producing the actual answer, and got truncated under RAGAS/
  RAGChecker's fixed token budgets -- RAGAS raised
  `LLMDidNotFinishException` on 4 of 5 metrics, and RAGChecker silently
  degenerated to all-zero scores (no error, just wrong numbers -- the
  dangerous kind of bug). Fixed by moving both to plain instruct models
  (`llama-3.3-70b-versatile` for RAGAS, `llama-3.1-8b-instant` for
  RAGChecker) and adding an explicit generous `max_tokens` to
  `groq_judge.py`'s client as a second line of defense. ARES's judge
  (`gpt-oss-20b`, also a reasoning model) was left alone -- it only
  regex-matches a `[[Yes/No]]` token, which survives truncation far better,
  and its live scores were already sane.
- **Ran one deliberate single-model comparison** (eval_id 2): every LLM role
  except ARES (which structurally requires "gpt" in the model name) forced
  onto one plain model. Quality held up identically to the per-subsystem
  spread; the expected difference in Groq 429 (rate-limit) counts didn't
  clearly show up at n=1 (28 vs. 23 -- noise, not signal, at this scale).
  Reverted `configs/models.py` back to the per-subsystem spread afterward.
- **Found and fixed a second, unrelated real bug** while building the
  detailed report below: `RAGTrace.configuration_snapshot["llm_model"]` was
  hardcoded from `configs.models.LLM_MODEL_NAME` (the HF-path constant)
  regardless of which provider/model actually generated the answer -- every
  trace claimed "Qwen/Qwen2.5-7B-Instruct" even when Groq's Llama-3.3-70B
  had actually run. Fixed in `src/rag_trace.py` by reading
  `generation_result.model_name` (the real model the `Generator` instance
  used) instead.
- **Built `scripts/generate_diagnostic_report.py`**: a detailed, per-trace
  HTML report distinct from `generate_comparison_report.py`'s aggregate
  markdown -- meant to be opened directly in a browser. Shows the case (Q&A),
  the full raw trace (prompt, retrieval scores), claim decomposition +
  verification (every atomic claim, its status, and its supporting evidence
  text), chunk-level provenance (source file/page, character offsets,
  configured chunk size/overlap, full chunk text), X-RAG's stage attribution,
  the root-cause reasoning chain, corrective actions, and -- when a matching
  row exists in `artifacts/benchmark_comparison/results.json` -- a
  side-by-side comparison against real RAGAS/RAGChecker/ARES scores for that
  same trace. Heavy sections (trace, claims, chunks) are tucked into native
  `<details>`/`<summary>` disclosure widgets, so the page opens compact and
  expands per-section on demand with zero JavaScript.

### Status
217 tests pass (213 core + this session's additions). Three live examples
now exist end-to-end under Groq, each with a full 4-framework comparison row
in `artifacts/benchmark_comparison/results.json` and a detailed HTML report
in `artifacts/diagnostic_reports/`. The full 40-example set has still not
been run (each example costs ~10-20 minutes, dominated by the CPU-only NLI
verification step) -- that remains the next step for a statistically
meaningful (rather than illustrative) comparison.
