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

### Problems Faced
1. **Extraction:** PDF readers failing on complex Indian Legal formats.
2. **Dense Search Blindspots:** Semantic embeddings fail hard at exact keyword matching (especially alphanumeric codes/sections).
3. **BM25 Skew:** Raw BM25 breaks down when querying with natural language sentences full of filler words, assigning extreme TF scores to irrelevant chunks.
4. **LLM formatting:** Using base-completion APIs on Instruct-tuned models completely ruins their ability to follow instructions.

### What We Plan To Do Next
- **RAGTrace Integration:** Start building the diagnostic builder that maps the `RetrievalResult` and `GenerationResult` against the `ChunkRegistry` to automatically flag hallucination, context drift, and retrieval drop-offs.
- **Eval Framework:** Introduce known failure conditions to the chunks and test if the diagnostic tool can catch them automatically.
