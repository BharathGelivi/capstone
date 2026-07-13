"""
Claim Verification Module (Phase 2.3).

Responsible for independently verifying atomic claims against retrieved evidence chunks
using a Natural Language Inference (NLI) model at the sentence level.
"""

import os
import json
import time
import uuid
import re
import logging
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional

from src.rag_trace import RAGTrace
from src.retriever import RetrievedChunk
from src.claim_decomposer import CandidateClaim, CandidateClaimSet
from src.config import (
    VERIFICATION_MODEL,
    VERIFICATION_BATCH_SIZE,
    ENTAILMENT_THRESHOLD,
    CONTRADICTION_THRESHOLD,
    PARTIAL_SUPPORT_THRESHOLD
)

try:
    from transformers import pipeline
except ImportError:
    pass

logger = logging.getLogger(__name__)

class VerificationStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNSUPPORTED = "UNSUPPORTED"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"


@dataclass
class EvidenceSentence:
    sentence_id: str
    chunk_id: str
    chunk_rank: int
    text: str
    entailment_score: float
    contradiction_score: float
    neutral_score: float


@dataclass
class VerificationResult:
    verification_id: str
    trace_id: str
    claim_id: str
    claim_text: str
    verification_status: VerificationStatus
    verification_reason: str
    confidence: float
    
    # Best single piece of evidence mapping
    best_chunk_id: Optional[str]
    best_chunk_rank: Optional[int]
    best_chunk_score: Optional[float]
    best_sentence_id: Optional[str]
    evidence_text: Optional[str]
    
    # Best evidence NLI scores
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    
    # Top 3 supporting sentences
    top_evidence: List[EvidenceSentence] = field(default_factory=list)
    
    verification_latency_ms: float = 0.0
    verification_model: str = VERIFICATION_MODEL


@dataclass
class VerificationSummary:
    trace_id: str
    total_claims: int
    supported_claims: int
    partially_supported_claims: int
    contradicted_claims: int
    unsupported_claims: int
    not_verifiable_claims: int
    average_entailment_score: float
    total_verification_latency_ms: float
    results: List[VerificationResult] = field(default_factory=list)

    def to_json(self) -> str:
        # Convert enums to strings for json serialization
        data = asdict(self)
        for res in data["results"]:
            res["verification_status"] = res["verification_status"].value
        return json.dumps(data, indent=4)


class ClaimVerifier:
    def __init__(self, model_name: str = VERIFICATION_MODEL):
        self.model_name = model_name
        logger.info(f"Loading NLI Verification Model: {self.model_name}")
        
        # Load zero-shot classification pipeline. 
        # Note: For deberta-v3-large-zeroshot-v2, the labels are entailment, neutral, contradiction.
        # Using standard text-classification for NLI formats: [Premise][SEP][Hypothesis]
        self.nli_pipeline = pipeline(
            "text-classification", 
            model=self.model_name, 
            return_all_scores=True,
            truncation=True
        )
            
    def _split_into_sentences(self, text: str) -> List[str]:
        """Simple regex-based sentence splitter."""
        # Split on . ! ? followed by a space and an uppercase letter, or end of string.
        # This is basic, but keeps dependencies low.
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def _determine_status_and_reason(self, entailment: float, contradiction: float, neutral: float) -> tuple[VerificationStatus, str]:
        """Resolves the scores into a final status and generates a reason."""
        if contradiction >= CONTRADICTION_THRESHOLD:
            return VerificationStatus.CONTRADICTED, f"Evidence explicitly contradicts the claim (Score: {contradiction:.2f})."
        elif entailment >= ENTAILMENT_THRESHOLD:
            return VerificationStatus.SUPPORTED, f"Evidence directly supports the claim (Score: {entailment:.2f})."
        elif entailment >= PARTIAL_SUPPORT_THRESHOLD:
            return VerificationStatus.PARTIALLY_SUPPORTED, f"Evidence provides partial/weak support (Score: {entailment:.2f})."
        elif neutral >= 0.8:
            return VerificationStatus.UNSUPPORTED, f"Evidence is neutral and does not support the claim (Neutral Score: {neutral:.2f})."
        else:
            return VerificationStatus.NOT_VERIFIABLE, "Evidence is insufficient for conclusive verification."

    def run_nli(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Runs the NLI pipeline on a single premise/hypothesis pair and returns scores."""
        # NLI Models usually expect premise and hypothesis combined or as text/text_pair
        # We use text=premise, text_pair=hypothesis in pipeline
        result = self.nli_pipeline({"text": premise, "text_pair": hypothesis})
        
        # Extract scores (handling varying label names across different NLI models)
        scores = {item['label'].lower(): item['score'] for item in result}
        
        entailment = scores.get("entailment", scores.get("label_0", 0.0))
        neutral = scores.get("neutral", scores.get("label_1", 0.0))
        contradiction = scores.get("contradiction", scores.get("label_2", 0.0))
        
        return {
            "entailment": entailment,
            "neutral": neutral,
            "contradiction": contradiction
        }

    def find_best_evidence(self, claim: CandidateClaim, chunks: List[RetrievedChunk]) -> List[EvidenceSentence]:
        """Finds and scores all sentences in chunks against a claim, returning all scored sentences."""
        all_scored_sentences: List[EvidenceSentence] = []
        
        for chunk in chunks:
            sentences = self._split_into_sentences(chunk.chunk_text)
            
            for i, sentence in enumerate(sentences):
                scores = self.run_nli(premise=sentence, hypothesis=claim.claim_text)
                
                s_id = f"{chunk.chunk_id}_s{i}"
                all_scored_sentences.append(
                    EvidenceSentence(
                        sentence_id=s_id,
                        chunk_id=chunk.chunk_id,
                        chunk_rank=chunk.rank,
                        text=sentence,
                        entailment_score=scores["entailment"],
                        contradiction_score=scores["contradiction"],
                        neutral_score=scores["neutral"]
                    )
                )
                
        # Sort sentences by entailment score descending
        all_scored_sentences.sort(key=lambda x: x.entailment_score, reverse=True)
        return all_scored_sentences

    def verify_claim(self, claim: CandidateClaim, chunks: List[RetrievedChunk]) -> VerificationResult:
        """Verifies a single claim against all retrieved chunks at the sentence level."""
        t_start = time.time()
        
        all_scored_sentences = self.find_best_evidence(claim, chunks)
        
        top_3 = all_scored_sentences[:3]
        best_sentence = top_3[0] if top_3 else None
        
        if best_sentence:
            status, reason = self._determine_status_and_reason(
                best_sentence.entailment_score,
                best_sentence.contradiction_score,
                best_sentence.neutral_score
            )
            # Use max of entailment or contradiction as confidence
            confidence = max(best_sentence.entailment_score, best_sentence.contradiction_score)
            
            res = VerificationResult(
                verification_id=str(uuid.uuid4()),
                trace_id=claim.trace_id,
                claim_id=claim.candidate_id,
                claim_text=claim.claim_text,
                verification_status=status,
                verification_reason=reason,
                confidence=confidence,
                best_chunk_id=best_sentence.chunk_id,
                best_chunk_rank=best_sentence.chunk_rank,
                best_chunk_score=None, # Needs to be passed if available in chunk metadata
                best_sentence_id=best_sentence.sentence_id,
                evidence_text=best_sentence.text,
                entailment_score=best_sentence.entailment_score,
                contradiction_score=best_sentence.contradiction_score,
                neutral_score=best_sentence.neutral_score,
                top_evidence=top_3,
                verification_latency_ms=(time.time() - t_start) * 1000
            )
        else:
            # Fallback if no evidence exists
            res = VerificationResult(
                verification_id=str(uuid.uuid4()),
                trace_id=claim.trace_id,
                claim_id=claim.candidate_id,
                claim_text=claim.claim_text,
                verification_status=VerificationStatus.UNSUPPORTED,
                verification_reason="No evidence sentences found.",
                confidence=0.0,
                best_chunk_id=None, best_chunk_rank=None, best_chunk_score=None,
                best_sentence_id=None, evidence_text=None,
                entailment_score=0.0, contradiction_score=0.0, neutral_score=0.0,
                verification_latency_ms=(time.time() - t_start) * 1000
            )
            
        logger.info(f"Verified claim '{claim.candidate_id}': {res.verification_status.value} (Latency: {res.verification_latency_ms:.1f}ms)")
        return res

    def verify_all(self, claim_set: CandidateClaimSet, trace_id: str, retrieved_chunks: List[RetrievedChunk]) -> VerificationSummary:
        """Verifies all claims in a CandidateClaimSet."""
        t_start = time.time()
        logger.info(f"Starting verification for trace {trace_id} ({claim_set.total_candidates} claims)")
        
        results = []
        for claim in claim_set.candidate_claims:
            res = self.verify_claim(claim, retrieved_chunks)
            results.append(res)
            
        # Aggregate statistics
        total = len(results)
        supported = sum(1 for r in results if r.verification_status == VerificationStatus.SUPPORTED)
        partial = sum(1 for r in results if r.verification_status == VerificationStatus.PARTIALLY_SUPPORTED)
        contradicted = sum(1 for r in results if r.verification_status == VerificationStatus.CONTRADICTED)
        unsupported = sum(1 for r in results if r.verification_status == VerificationStatus.UNSUPPORTED)
        not_verif = sum(1 for r in results if r.verification_status == VerificationStatus.NOT_VERIFIABLE)
        
        avg_entailment = sum(r.entailment_score for r in results) / total if total > 0 else 0.0
        total_latency = (time.time() - t_start) * 1000
        
        summary = VerificationSummary(
            trace_id=trace_id,
            total_claims=total,
            supported_claims=supported,
            partially_supported_claims=partial,
            contradicted_claims=contradicted,
            unsupported_claims=unsupported,
            not_verifiable_claims=not_verif,
            average_entailment_score=avg_entailment,
            total_verification_latency_ms=total_latency,
            results=results
        )
        
        logger.info(f"Completed verification for trace {trace_id}. "
                    f"Supported: {supported}/{total}, Contradicted: {contradicted}/{total}")
        return summary

    def save_artifacts(self, summary: VerificationSummary):
        """Saves the verification summary to a JSON file."""
        dir_path = os.path.join("artifacts", "verification")
        os.makedirs(dir_path, exist_ok=True)
        
        filepath = os.path.join(dir_path, f"TRACE_{summary.trace_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(summary.to_json())
        logger.info(f"Saved verification artifact to {filepath}")
