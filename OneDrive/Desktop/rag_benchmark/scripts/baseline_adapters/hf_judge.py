"""
Shared helper for pointing baseline evaluation frameworks' LLM judges at
Hugging Face's OpenAI-compatible router (https://router.huggingface.co/v1),
reusing this project's existing HF_TOKEN setup rather than requiring a
separate OpenAI/Bedrock key for the comparison scripts.
"""

import os

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"


def build_hf_chat_openai(model: str, temperature: float = 0.0):
    """Builds a LangChain ChatOpenAI client pointed at HF's OpenAI-compatible router."""
    from langchain_openai import ChatOpenAI

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN must be set to use the HF-hosted judge LLM for baseline comparisons.")
    return ChatOpenAI(model=model, base_url=HF_ROUTER_BASE_URL, api_key=token, temperature=temperature)
