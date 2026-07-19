# LLM and Model Configurations

# The name of the embedding model to use from HuggingFace.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Reranker model
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

# The Hugging Face model to use for answer synthesis and claim decomposition
# (only used when LLM_PROVIDER == "huggingface" -- see below).
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Generation temperature
LLM_TEMPERATURE = 0.1

# Maximum number of tokens to generate in the response
LLM_MAX_TOKENS = 1024

# Which LLM backend powers generation, claim decomposition, and the baseline
# judges: "groq" (free tier, https://console.groq.com) or "huggingface"
# (HF Inference Providers -- needs purchased credits/PRO once the free
# monthly allowance is exhausted). Switch back to "huggingface" once HF
# credits are available again; no other code changes needed.
LLM_PROVIDER = "groq"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Groq's free tier rate-limits per model independently, so each pipeline
# stage/judge below uses a different model -- spreads load across separate
# RPM/RPD buckets instead of all contending for the same one. (A one-off
# single-model comparison run -- eval_id 2, 2026-07-19 -- temporarily set
# every role to llama-3.3-70b-versatile to see how the pipeline behaves
# without per-subsystem spread; see docs/RESEARCH_LOG.md for that result.)
GROQ_GENERATION_MODEL = "llama-3.3-70b-versatile"       # X-RAG answer synthesis
GROQ_CLAIM_DECOMPOSER_MODEL = "llama-3.1-8b-instant"    # X-RAG claim decomposition
# RAGAS and RAGChecker both need clean, immediately-parseable JSON/score
# output from a single completion under a fixed token budget. "Reasoning"
# models (qwen3.6, gpt-oss-*) spend that budget on an internal <think> block
# first and got truncated before finishing (RAGAS: LLMDidNotFinishException;
# RAGChecker: silently degenerated to all-zero scores) -- verified live, not
# a guess. Plain instruct models don't have this failure mode, so RAGAS and
# RAGChecker share the two non-reasoning models with generation/decomposition
# instead of each getting a dedicated reasoning-model bucket.
GROQ_RAGAS_JUDGE_MODEL = "llama-3.3-70b-versatile"      # RAGAS's LLM judge
GROQ_RAGCHECKER_JUDGE_MODEL = "llama-3.1-8b-instant"    # RAGChecker's extractor/checker
GROQ_ARES_JUDGE_MODEL = "openai/gpt-oss-20b"            # ARES's ues_idp judge (name
# must contain "gpt": ARES routes to its "gpt" scoring path via a literal
# `"gpt" in model_choice` substring check, then sends that same string as the
# API model= parameter -- see scripts/baseline_adapters/ares_worker.py. This
# IS a reasoning model too, but ARES only regex-matches a "[[Yes/No]]" token
# out of the response, which survives truncation far better than RAGAS/
# RAGChecker's structured JSON parsing -- confirmed live (sane, non-zero
# scores), so left as-is.

# Verification model
VERIFICATION_MODEL = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
VERIFICATION_BATCH_SIZE = 8
