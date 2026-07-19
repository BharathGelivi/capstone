"""
Converts the shared ResolvedExample intermediate format (see common.py) into
the RAGResults format the installed `ragchecker` package expects.

Schema note (verified against the installed package's container classes, not
assumed): `ragchecker.container.RAGResult` requires `query_id`/`query`/
`gt_answer`/`response`, and `gt_answer` has no default -- unlike ragas,
RAGChecker cannot score an example at all without a gold answer, so examples
with no gold answer are skipped here rather than included with a null
reference.

Install note: `ragchecker` needs its own Python 3.10 venv (see
requirements-eval-ragchecker.txt) -- this module is meant to be run under
that venv's interpreter, not this project's main venv, and therefore works
off ResolvedExample (plain strings) rather than RAGTrace/ChunkRegistry
directly (those need llama-index, which isn't installed in that venv).
"""

import logging
import os
from typing import List

from scripts.baseline_adapters.common import ResolvedExample

logger = logging.getLogger(__name__)


def to_ragchecker_results(examples: List[ResolvedExample]):
    """
    Builds a ragchecker RAGResults object from ResolvedExamples. Examples
    without a gold answer are skipped (logged), since RAGChecker requires one
    for every example.
    """
    from ragchecker import RAGResults
    from ragchecker.container import RAGResult, RetrievedDoc

    results = []
    for example in examples:
        if example.gold_answer is None:
            logger.warning(f"Skipping trace {example.trace_id} for RAGChecker: no gold answer available.")
            continue

        results.append(RAGResult(
            query_id=example.trace_id,
            query=example.question,
            gt_answer=example.gold_answer,
            response=example.answer,
            retrieved_context=[RetrievedDoc(doc_id=f"{example.trace_id}_c{i}", text=text) for i, text in enumerate(example.contexts)],
        ))

    return RAGResults(results=results)


def build_ragchecker(model: str = None, batch_size: int = 4):
    """
    Builds a RAGChecker instance whose extractor/checker LLM calls route through
    an OpenAI-compatible endpoint (litellm's `openai/<model>` custom-endpoint
    convention) -- Groq's free tier by default (configs.models.LLM_PROVIDER),
    or HF's router once HF credits are available again. No Bedrock/OpenAI key
    required either way.

    Note: this module runs under venv_eval_ragchecker's own Python 3.10
    interpreter (see module docstring), which does NOT have configs/ installed
    -- provider selection here is driven by GROQ_API_KEY/HF_TOKEN env vars
    directly rather than importing configs.models.
    """
    from ragchecker import RAGChecker

    groq_key = os.environ.get("GROQ_API_KEY")
    hf_token = os.environ.get("HF_TOKEN")

    if groq_key:
        api_base = "https://api.groq.com/openai/v1"
        api_key = groq_key
        # Plain instruct model, not a "reasoning" one (qwen3.6/gpt-oss-*) --
        # those spend the token budget on an internal <think> block and got
        # truncated before finishing RAGChecker's structured JSON output,
        # silently degenerating to all-zero scores (verified live).
        model = model or "llama-3.1-8b-instant"
    elif hf_token:
        api_base = "https://router.huggingface.co/v1"
        api_key = hf_token
        model = model or "Qwen/Qwen2.5-7B-Instruct"
    else:
        raise ValueError("GROQ_API_KEY or HF_TOKEN must be set to use a judge LLM for RAGChecker.")

    litellm_model = f"openai/{model}"
    return RAGChecker(
        extractor_name=litellm_model,
        checker_name=litellm_model,
        extractor_api_base=api_base,
        checker_api_base=api_base,
        openai_api_key=api_key,
        batch_size_extractor=batch_size,
        batch_size_checker=batch_size,
    )
