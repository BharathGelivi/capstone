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
