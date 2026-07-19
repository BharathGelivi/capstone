"""
Generator Module.
Responsible exclusively for synthesizing answers using a Large Language Model
based on the results provided by the Retriever.
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from llama_index.llms.groq import Groq
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.llms import ChatMessage, MessageRole
from src.retriever import RetrievalResult
from configs.models import (
    LLM_MODEL_NAME, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_PROVIDER, GROQ_GENERATION_MODEL,
)
from configs.prompts import DEFAULT_SYSTEM_INSTRUCTIONS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GenerationResult:
    """
    A comprehensive result object capturing the entire generation event.
    Provides complete transparency for future diagnostics, tracking exactly
    what the LLM saw and what it produced.
    """
    question: str
    generated_answer: str
    prompt: str
    prompt_length: int
    model_name: str
    temperature: float
    max_tokens: int
    generation_time: float
    retrieved_chunk_ids: List[str]
    generation_metadata: Dict[str, Any]


class PromptBuilder:
    """
    Constructs the prompt string clearly separating context, instructions, and the question.
    Isolated from the Generator to allow easy prompt experimentation later.
    """
    
    @staticmethod
    def build_messages(question: str, retrieved_chunks: List[Any], system_instructions: Optional[str] = None) -> List[ChatMessage]:
        """
        Builds the chat messages array (System and User).
        """
        # 1. Instructions
        instructions = system_instructions or DEFAULT_SYSTEM_INSTRUCTIONS

        # 2. Context
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            context_parts.append(f"--- Context chunk {i} [Chunk-ID: {chunk.chunk_id}] ---\n{chunk.chunk_text}\n")

        context = "\n".join(context_parts)
        if not context.strip():
            context = "No relevant context found."
            
        # 3. Question
        user_message = f"{context}\n\nQuestion: {question}"
        
        return [
            ChatMessage(role=MessageRole.SYSTEM, content=instructions),
            ChatMessage(role=MessageRole.USER, content=user_message)
        ]


class Generator:
    """
    Executes the generation phase using the configured LLM.
    """
    def __init__(self, model_name: Optional[str] = None, temperature: float = LLM_TEMPERATURE, max_tokens: int = LLM_MAX_TOKENS, system_instructions: Optional[str] = None):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_instructions = system_instructions

        if LLM_PROVIDER == "groq":
            self.model_name = model_name or GROQ_GENERATION_MODEL
            logger.info(f"Initializing Generator with Groq model {self.model_name}")
            self.llm = Groq(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=os.environ.get("GROQ_API_KEY"),
            )
        else:
            self.model_name = model_name or LLM_MODEL_NAME
            logger.info(f"Initializing Generator with HF model {self.model_name}")
            # Requires HF_TOKEN environment variable.
            self.llm = HuggingFaceInferenceAPI(
                model_name=self.model_name,
                temperature=self.temperature,
                num_output=self.max_tokens,  # NOT max_new_tokens: that kwarg doesn't exist on this
                # class and is silently dropped, leaving the 256-token library default in place.
                token=os.environ.get("HF_TOKEN")
            )
        
    def generate(self, retrieval_result: RetrievalResult) -> GenerationResult:
        """
        Consumes a RetrievalResult and produces a GenerationResult.
        """
        logger.info(f"Initiating generation for question: '{retrieval_result.question}'")
        start_time = time.time()
        
        # Build the prompt using the separate builder
        messages = PromptBuilder.build_messages(
            question=retrieval_result.question,
            retrieved_chunks=retrieval_result.retrieved_chunks,
            system_instructions=self.system_instructions
        )
        # Store a string representation for logging/metadata
        prompt_str = "\n".join([f"[{m.role.value.upper()}]: {m.content}" for m in messages])
        
        try:
            # Pass the constructed chat messages to the LLM
            response = self.llm.chat(messages)
            generated_answer = str(response.message.content)
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            generated_answer = f"Error generating answer: {e}"
            
        generation_time = time.time() - start_time
        
        # Construct the GenerationResult capturing all diagnostic state
        result = GenerationResult(
            question=retrieval_result.question,
            generated_answer=generated_answer,
            prompt=prompt_str,
            prompt_length=len(prompt_str),
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            generation_time=generation_time,
            retrieved_chunk_ids=retrieval_result.retrieved_chunk_ids,
            generation_metadata={
                "retrieval_time": retrieval_result.retrieval_time,
                "top_k_used": retrieval_result.top_k
            }
        )
        
        logger.info(f"Generation complete in {generation_time:.3f}s.")
        return result
