"""
Claim Decomposer Module.

Responsible for decomposing a generated answer from a RAGTrace into atomic, verifiable facts (candidate claims).
"""

import os
import json
import uuid
import re
import time
import logging
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from llama_index.llms.gemini import Gemini
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.llms import ChatMessage, MessageRole

from src.rag_trace import RAGTrace
from configs.pipeline import CLAIM_DECOMPOSER_PROMPT_VERSION, CLAIM_DECOMPOSER_MAX_TOKENS

logger = logging.getLogger(__name__)

@dataclass
class CandidateClaim:
    candidate_id: str
    trace_id: str
    claim_text: str
    source_sentence: str
    sentence_id: str
    claim_index: int
    character_start: int
    character_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CandidateClaimSet:
    trace_id: str
    candidate_claims: List[CandidateClaim] = field(default_factory=list)
    total_candidates: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_claim(self, claim: CandidateClaim):
        self.candidate_claims.append(claim)
        self.total_candidates += 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "CandidateClaimSet":
        data = json.loads(json_str)
        claims = [CandidateClaim(**c) for c in data.pop("candidate_claims", [])]
        return cls(candidate_claims=claims, **data)


class JSONRecoveryError(Exception):
    pass


class ClaimDecomposer:
    def __init__(self, debug: Optional[bool] = None, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        if debug is None:
            self.debug = os.environ.get("CLAIM_DECOMPOSER_DEBUG", "False").lower() == "true"
        else:
            self.debug = debug
            
        self.model_name = model_name
        if model_name == "gemini":
            self.llm = Gemini(model="models/gemini-1.5-pro-latest", temperature=0.1)
        else:
            self.llm = HuggingFaceInferenceAPI(
                model_name=model_name,
                temperature=0.1,
                max_new_tokens=CLAIM_DECOMPOSER_MAX_TOKENS,
                token=os.environ.get("HF_TOKEN")
            )

    def decompose(self, trace: RAGTrace) -> CandidateClaimSet:
        logger.info(f"Decomposing answer for trace_id: {trace.trace_id}")
        answer = trace.generated_answer
        
        t0 = time.time()

        instructions = (
            "You are an expert fact extractor. Your task is to decompose the following text into atomic factual claims.\n\n"
            "Definition:\n"
            "An atomic claim is the smallest semantically complete factual assertion that can be independently verified.\n\n"
            "Rules:\n"
            "Rule 1: One fact per claim.\n"
            "Rule 2: Claims must be semantically complete.\n"
            "Rule 3: Claims must be independently verifiable.\n"
            "Rule 4: Do not infer information.\n"
            "Rule 5: Do not generate opinions.\n"
            "Rule 6: Do not rewrite meaning.\n"
            "Rule 7: Don't merge independently verifiable facts.\n"
            "Rule 8: Don't invent claims not explicitly stated.\n\n"
        )
        
        output_format = (
            "Return the output ONLY as a valid JSON array of objects. Do not include markdown formatting, explanations, code fences, or any additional text.\n"
            "Example format:\n"
            "[\n"
            "  {\n"
            '    "claim_text": "...",\n'
            '    "sentence_id": "S001"\n'
            "  }\n"
            "]\n"
        )
        
        prompt = f"{instructions}{output_format}\nText to decompose:\n{answer}"

        # Track latencies
        generation_latency = 0.0
        parsing_latency = 0.0
        recovery_latency = 0.0
        retry_latency = 0.0
        
        # Diagnostics tracking
        parsing_error = None
        recovery_attempt = False
        retry_attempt = False
        success = False

        # First LLM call
        t_gen_start = time.time()
        response_str = self._call_llm(prompt)
        generation_latency = time.time() - t_gen_start

        if self.debug:
            self._save_debug_artifacts(trace.trace_id, prompt, response_str)

        # Parsing attempt
        t_parse_start = time.time()
        try:
            raw_claims = self._robust_json_parse(response_str)
            success = True
            parsing_latency = time.time() - t_parse_start
        except JSONRecoveryError as e:
            parsing_latency = time.time() - t_parse_start
            parsing_error = str(e)
            recovery_attempt = True
            retry_attempt = True
            logger.warning(f"Initial parse and recovery failed. Initiating retry. Error: {e}")
            
            # Retry attempt
            retry_prompt = (
                f"The previous response was not valid JSON.\n"
                f"Return ONLY the corrected JSON.\n"
                f"Do not change the content.\n"
                f"Do not add explanations.\n\n"
                f"Original Response:\n{response_str}"
            )
            
            t_retry_start = time.time()
            retry_response_str = self._call_llm(retry_prompt)
            retry_latency = time.time() - t_retry_start
            
            t_retry_parse_start = time.time()
            try:
                raw_claims = self._robust_json_parse(retry_response_str)
                success = True
                response_str = retry_response_str # update raw response for diagnostics
            except JSONRecoveryError as e2:
                parsing_error = str(e2)
                logger.error(f"Retry parsing failed: {e2}")
                raw_claims = []
            
            recovery_latency = time.time() - t_retry_parse_start

        total_decomposition_latency = time.time() - t0

        logger.info(
            f"Latencies - Generation: {generation_latency:.3f}s, "
            f"Parsing: {parsing_latency:.3f}s, "
            f"Retry: {retry_latency:.3f}s, "
            f"Recovery: {recovery_latency:.3f}s, "
            f"Total: {total_decomposition_latency:.3f}s"
        )

        claim_set = CandidateClaimSet(
            trace_id=trace.trace_id,
            metadata={"CLAIM_DECOMPOSER_PROMPT_VERSION": CLAIM_DECOMPOSER_PROMPT_VERSION}
        )

        unique_sentences = set()
        
        for i, raw_claim in enumerate(raw_claims):
            claim_text = raw_claim.get("claim_text", "")
            sentence_id = raw_claim.get("sentence_id", "")
            
            start_idx = answer.find(claim_text)
            end_idx = start_idx + len(claim_text) if start_idx != -1 else -1

            if sentence_id:
                unique_sentences.add(sentence_id)

            meta = {}
            candidate = CandidateClaim(
                candidate_id=str(uuid.uuid4()),
                trace_id=trace.trace_id,
                claim_text=claim_text,
                source_sentence="", # Removed from schema, kept in model for backward compatibility
                sentence_id=sentence_id,
                claim_index=i,
                character_start=start_idx,
                character_end=end_idx,
                metadata=meta
            )
            claim_set.add_claim(candidate)

        # Calculate statistics
        total_sentences = len(unique_sentences)
        total_claims = claim_set.total_candidates
        avg_claims = (total_claims / total_sentences) if total_sentences > 0 else 0.0

        claim_set.metadata.update({
            "total_sentences": total_sentences,
            "total_candidate_claims": total_claims,
            "average_claims_per_sentence": round(avg_claims, 2),
            "diagnostics": {
                "trace_id": trace.trace_id,
                "raw_response": response_str,
                "parsing_error": parsing_error,
                "recovery_attempt": recovery_attempt,
                "retry_attempt": retry_attempt,
                "success": success,
                "latencies": {
                    "generation_latency": round(generation_latency, 3),
                    "parsing_latency": round(parsing_latency, 3),
                    "recovery_latency": round(recovery_latency, 3),
                    "retry_latency": round(retry_latency, 3),
                    "total_decomposition_latency": round(total_decomposition_latency, 3)
                }
            }
        })

        return claim_set

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.llm.complete(prompt)
            return str(response.text)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ""

    def _robust_json_parse(self, text: str) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            raise JSONRecoveryError("Empty string provided to JSON parser.")
            
        text = text.strip()

        # 1. Direct try
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        # 3. Regex extraction of largest array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            array_text = match.group(0).strip()
            try:
                return json.loads(array_text)
            except json.JSONDecodeError:
                text = array_text

        # 4. Bracket balancing
        open_brackets = text.count('[')
        close_brackets = text.count(']')
        open_braces = text.count('{')
        close_braces = text.count('}')
        
        balanced_text = text
        
        if open_braces > close_braces:
            balanced_text += '}' * (open_braces - close_braces)
            
        if open_brackets > close_brackets:
            balanced_text += ']' * (open_brackets - close_brackets)
            
        try:
            return json.loads(balanced_text)
        except json.JSONDecodeError as e:
            raise JSONRecoveryError(f"Failed to recover JSON: {e}")

    def _save_debug_artifacts(self, trace_id: str, prompt: str, response: str):
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        dir_path = os.path.join("artifacts", "debug", date_str)
        os.makedirs(dir_path, exist_ok=True)
        
        prompt_path = os.path.join(dir_path, f"prompt_{trace_id}.txt")
        with open(prompt_path, 'w', encoding='utf-8') as f:
            f.write(prompt)
            
        response_path = os.path.join(dir_path, f"response_{trace_id}.json")
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(response)
            
        logger.info(f"Saved debug artifacts for trace {trace_id}")
