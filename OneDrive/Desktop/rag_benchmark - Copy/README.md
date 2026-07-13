# RAG Diagnostics and Evaluation

## 1. Project Overview
This project is a research initiative focused on building a diagnostic and evaluation framework for Retrieval-Augmented Generation (RAG) systems. A benchmark RAG pipeline is being constructed sequentially to generate inputs (chunks, embeddings, retrieval outputs) that will later be analyzed by our diagnostic tools.

## 2. Research Motivation
Most RAG implementations are treated as "black boxes" where documents go in and answers come out. When a RAG system fails, it is incredibly difficult to determine whether the failure was caused by poor chunking, faulty embeddings, incorrect retrieval, or LLM hallucination.

## 3. Research Gap
Current evaluation tools often only measure the final output (e.g., using LLM-as-a-judge). There is a significant gap in granular, step-by-step diagnostic tracking that allows researchers to pinpoint the exact module and exact line of text where the pipeline broke down.

## 4. Proposed Framework
We propose a transparent, trackable framework that maintains canonical registries parallel to the standard RAG pipeline. By detaching metadata and text origin tracking from the vector database, we can perform robust evidence mapping and failure attribution across different embedding models and retrieval strategies.

## 5. Repository Structure
- `data/`: Raw documents for ingestion.
- `db/`: Local storage for the vector database (ChromaDB).
- `src/`: Modular source code for the pipeline.
- `artifacts/`: Serialized outputs for reproducibility.
- `docs/`: Research logs and documentation.

## 6. Current Architecture
```text
Legal PDFs (or general text)
      ↓
Document Loader (ingestion.py)
      ↓
Chunk Engine (chunk_engine.py)
      ↓
Chunk Registry (chunk_registry.py)
      ↓
Embedding Engine (embedding_engine.py)
      ↓
Vector Store Interface (vector_store.py)
      ↓
ChromaVectorStore
      ↓
ChromaDB
      ↓
Retriever (retriever.py)
  [Dense: ChromaDB] + [Sparse: BM25]
      ↓
RRF Fusion (Top-20 candidates)
      ↓
Cross-Encoder Reranker (Top-5 candidates)
      ↓
RetrievalResult
      ↓
Generator (generator.py)
      ↓
GenerationResult
      ↓
RAGTrace (rag_trace.py)
      ↓
Claim Decomposer (claim_decomposer.py)
      ↓
CandidateClaimSet
      ↓
Claim Representation Framework (claims.py)
      ↓
ClaimSet
      ↓
(Diagnostic Framework / Claim Engine - Pending)
```

## 7. Current Progress
The core RAG benchmark pipeline is functionally complete! The system can load documents, chunk, register, embed, store, retrieve, and ultimately generate answers using the Gemini LLM. Every step of this process outputs deeply introspective data models (`ChunkRecord`, `RetrievalResult`, `GenerationResult`) that are now ready to be consumed by the upcoming diagnostic framework (RAGTrace).

## 8. Completed Modules
- [x] **Document Loader**: Reads local PDFs into LlamaIndex Document objects.
- [x] **Chunk Engine**: Splits documents into semantically appropriate nodes.
- [x] **Chunk Registry**: Maintains an independent, serializable JSON registry of all chunks.
- [x] **Embedding Engine**: Converts text chunks into numerical vector embeddings.
- [x] **Vector Store**: A decoupled ChromaDB implementation to store vectors and metadata.
- [x] **Retriever**: Hybrid Search combining Semantic embeddings (Dense) and BM25 (Sparse) via Reciprocal Rank Fusion (RRF). Uses a Cross-Encoder (`BAAI/bge-reranker-base`) to rerank Top-20 candidates down to Top-5.
- [x] **Generator**: Synthesizes answers using Gemini while capturing exact prompt states and generation metadata.
- [x] **RAGTrace Builder**: Standardized execution recorder that outputs JSON traces containing configuration snapshots, chunk references, and generation metadata for downstream diagnostics.
- [x] **Claim Representation Framework**: Core data structures (`Claim`, `ClaimSet`) separating atomic facts from evaluation metrics, enabling serialized offline diagnostics.

## 9. Upcoming Modules
- [ ] **Diagnostic Framework (Claim Engine)**: Evidence mapping, claim evaluation, and failure attribution using the `RAGTrace` artifact.

## 10. Experimental Roadmap
1. Establish the baseline benchmark RAG pipeline.
2. Ingest a controlled dataset with known failure conditions.
3. Run the diagnostic tools against the pipeline's outputs.
4. Evaluate the framework's ability to accurately attribute failure causes.

## 11. How to Run

### Step 1: Ingestion Pipeline
To ingest your documents, chunk them, and store them in ChromaDB, run:
```bash
python run_pipeline.py
```
*Note: Make sure your PDF documents are placed in the `data/` folder and you have run `pip install -r requirements.txt` before running.*

### Step 2: Querying the Database
To query the database and get an AI-generated answer, you need a Hugging Face API token (because we are using Llama-3 via Hugging Face Inference API). 
Set the environment variable and run the query script:

```powershell
# Windows PowerShell
$env:HF_TOKEN="your_huggingface_token_here"
python query.py "Your question here?"
```

```bash
# Mac/Linux
export HF_TOKEN="your_huggingface_token_here"
python query.py "Your question here?"
```

## 12. Research Notes
*For detailed implementation logs, decisions, and trade-offs, please refer to the `docs/RESEARCH_LOG.md`.*
