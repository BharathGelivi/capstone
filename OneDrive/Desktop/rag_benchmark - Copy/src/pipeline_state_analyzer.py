import os
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from src.rag_trace import RAGTrace
from src.claims import ClaimSet
from src.claim_verifier import VerificationSummary, VerificationStatus

class PipelineStage(str, Enum):
    CORPUS = "CORPUS"
    RETRIEVER = "RETRIEVER"
    CHUNKING = "CHUNKING"
    GENERATOR = "GENERATOR"
    GROUNDING = "GROUNDING"

class PipelineStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"

@dataclass
class PipelineState:
    stage: PipelineStage
    status: PipelineStatus
    observation: str
    confidence: float
    supporting_claim_ids: List[str] = field(default_factory=list)
    supporting_chunk_ids: List[str] = field(default_factory=list)
    supporting_verification_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineStateMatrix:
    trace_id: str
    artifact_version: str = "1.0"
    pipeline_states: List[PipelineState] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def get(self, stage: PipelineStage) -> Optional[PipelineState]:
        for state in self.pipeline_states:
            if state.stage == stage:
                return state
        return None

    def summary(self) -> Dict[str, str]:
        return {state.stage.value: state.status.value for state in self.pipeline_states}

    def to_json(self) -> str:
        data = asdict(self)
        for state in data["pipeline_states"]:
            state["stage"] = state["stage"].value
            state["status"] = state["status"].value
        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, data_str: str) -> 'PipelineStateMatrix':
        data = json.loads(data_str)
        states = []
        for state_data in data.get("pipeline_states", []):
            state_data["stage"] = PipelineStage(state_data["stage"])
            state_data["status"] = PipelineStatus(state_data["status"])
            states.append(PipelineState(**state_data))
        data["pipeline_states"] = states
        return cls(**data)

    def save(self, base_dir: str = "artifacts/pipeline_state_matrix") -> str:
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"TRACE_{self.trace_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        return filepath

    @classmethod
    def load(cls, filepath: str) -> 'PipelineStateMatrix':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

class PipelineStateAnalyzer:
    def __init__(self, retrieval_score_threshold: float = 0.5):
        self.retrieval_score_threshold = retrieval_score_threshold

    def analyze(self, trace: RAGTrace, claim_set: ClaimSet, verification: VerificationSummary) -> PipelineStateMatrix:
        states = []
        
        supported_verifications = [
            v for v in verification.results 
            if v.verification_status in (VerificationStatus.SUPPORTED, VerificationStatus.PARTIALLY_SUPPORTED)
        ]
        unsupported_verifications = [
            v for v in verification.results 
            if v.verification_status == VerificationStatus.UNSUPPORTED
        ]
        
        has_supported_claim = len(supported_verifications) > 0
        all_unsupported = len(unsupported_verifications) == len(verification.results) and len(verification.results) > 0
        some_unsupported = len(unsupported_verifications) > 0
        
        max_score = 0.0
        if trace.retrieved_chunk_references:
            max_score = max((chunk.get("similarity_score", 0.0) for chunk in trace.retrieved_chunk_references), default=0.0)
            
        supporting_claim_ids = [v.claim_id for v in supported_verifications]
        supporting_chunk_ids = [v.best_chunk_id for v in supported_verifications if v.best_chunk_id]
        supporting_verification_ids = [v.verification_id for v in supported_verifications]
        
        # 1. Corpus
        states.append(PipelineState(
            stage=PipelineStage.CORPUS,
            status=PipelineStatus.UNKNOWN,
            observation="Corpus statistics are currently unobservable.",
            confidence=1.0,
            supporting_claim_ids=[],
            supporting_chunk_ids=[],
            supporting_verification_ids=[]
        ))
        
        # 2. Retriever
        if has_supported_claim:
            retriever_status = PipelineStatus.PASS
            retriever_obs = "At least one retrieved chunk provides supporting evidence for a verified claim."
            retriever_conf = 0.95
        elif all_unsupported and max_score < self.retrieval_score_threshold:
            retriever_status = PipelineStatus.FAIL
            retriever_obs = "No retrieved chunks provide support, and retrieval scores strongly indicate a lack of useful evidence."
            retriever_conf = 0.85
        else:
            retriever_status = PipelineStatus.UNKNOWN
            retriever_obs = "Retrieval evidence does not definitively confirm or refute pipeline success."
            retriever_conf = 0.5

        states.append(PipelineState(
            stage=PipelineStage.RETRIEVER,
            status=retriever_status,
            observation=retriever_obs,
            confidence=retriever_conf,
            supporting_claim_ids=supporting_claim_ids,
            supporting_chunk_ids=supporting_chunk_ids,
            supporting_verification_ids=supporting_verification_ids
        ))
        
        # 3. Chunking
        states.append(PipelineState(
            stage=PipelineStage.CHUNKING,
            status=PipelineStatus.UNKNOWN,
            observation="Deterministic chunk-boundary evidence is currently unobservable.",
            confidence=1.0,
            supporting_claim_ids=[],
            supporting_chunk_ids=[],
            supporting_verification_ids=[]
        ))
        
        # 4. Generator
        if some_unsupported and max_score >= self.retrieval_score_threshold:
            gen_status = PipelineStatus.FAIL
            gen_obs = f"{len(unsupported_verifications)} generated claims remain unsupported despite high retrieval scores."
            gen_conf = 0.85
        elif some_unsupported:
            gen_status = PipelineStatus.UNKNOWN
            gen_obs = f"{len(unsupported_verifications)} generated claims are unsupported, but retrieval quality is ambiguous."
            gen_conf = 0.5
        else:
            gen_status = PipelineStatus.PASS
            gen_obs = "All generated claims are fully supported by evidence."
            gen_conf = 0.95
            
        states.append(PipelineState(
            stage=PipelineStage.GENERATOR,
            status=gen_status,
            observation=gen_obs,
            confidence=gen_conf,
            supporting_claim_ids=supporting_claim_ids,
            supporting_chunk_ids=supporting_chunk_ids,
            supporting_verification_ids=supporting_verification_ids
        ))
        
        # 5. Grounding
        if not some_unsupported and len(verification.results) > 0:
            grounding_status = PipelineStatus.PASS
            grounding_obs = "Every verified claim is supported by retrieved evidence."
            grounding_conf = 0.95
        else:
            grounding_status = PipelineStatus.FAIL
            grounding_obs = f"One or more claims ({len(unsupported_verifications)}) remain unsupported by evidence."
            grounding_conf = 0.95

        states.append(PipelineState(
            stage=PipelineStage.GROUNDING,
            status=grounding_status,
            observation=grounding_obs,
            confidence=grounding_conf,
            supporting_claim_ids=supporting_claim_ids,
            supporting_chunk_ids=supporting_chunk_ids,
            supporting_verification_ids=supporting_verification_ids
        ))

        return PipelineStateMatrix(
            trace_id=trace.trace_id,
            pipeline_states=states
        )
