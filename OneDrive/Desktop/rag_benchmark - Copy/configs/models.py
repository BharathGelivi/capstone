# LLM and Model Configurations

# The name of the embedding model to use from HuggingFace.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Reranker model
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# The Hugging Face model to use for answer synthesis
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Generation temperature
LLM_TEMPERATURE = 0.1

# Maximum number of tokens to generate in the response
LLM_MAX_TOKENS = 1024

# Verification model
VERIFICATION_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
VERIFICATION_BATCH_SIZE = 8
