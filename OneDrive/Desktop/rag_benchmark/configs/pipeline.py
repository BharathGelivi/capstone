# Pipeline configurations

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Vector Store
CHROMA_PERSIST_DIR = "./db/chroma"
CHROMA_COLLECTION_NAME = "rag_benchmark_collection"

# Retrieval
RETRIEVAL_TOP_K = 5
RERANKER_TOP_N = 5

# Claim Decomposer
CLAIM_DECOMPOSER_PROMPT_VERSION = "1.0"
CLAIM_DECOMPOSER_MAX_TOKENS = 4096
