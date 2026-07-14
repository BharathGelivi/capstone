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
