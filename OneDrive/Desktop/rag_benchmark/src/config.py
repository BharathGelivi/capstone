"""
Configuration settings for the RAG benchmark project.
This file holds configurable parameters to keep the codebase modular and avoid hardcoding values.
"""

# ------------------------------------------------------------------
# Chunking Configuration
# ------------------------------------------------------------------

# The maximum number of tokens/characters per chunk. 
# 512 is a common default for many dense embedding models (like BAAI/bge-small-en-v1.5)
CHUNK_SIZE = 512

# The number of overlapping tokens/characters between consecutive chunks.
# This ensures that concepts spanning across a chunk boundary are not lost.
CHUNK_OVERLAP = 50

# ------------------------------------------------------------------
# Embedding Configuration
# ------------------------------------------------------------------

# The name of the embedding model to use from HuggingFace.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ------------------------------------------------------------------
# Vector Store Configuration
# ------------------------------------------------------------------

# Directory to persist ChromaDB files locally.
CHROMA_PERSIST_DIR = "./db/chroma"

# Name of the ChromaDB collection.
CHROMA_COLLECTION_NAME = "rag_benchmark_collection"

# ------------------------------------------------------------------
# Retrieval Configuration
# ------------------------------------------------------------------

# The number of chunks to retrieve during a search (Top-K)
RETRIEVAL_TOP_K = 5

# ------------------------------------------------------------------
# Generation Configuration
# ------------------------------------------------------------------

# The Hugging Face model to use for answer synthesis (Llama 3 is excellent for this)
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Generation temperature (0.1 for most deterministic, predictable output in benchmarks)
# Note: Hugging Face API requires temperature > 0.
LLM_TEMPERATURE = 0.1

# Maximum number of tokens to generate in the response
LLM_MAX_TOKENS = 1024



