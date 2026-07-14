import os
import json
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from src.pipeline_state_analyzer import PipelineStateMatrix, PipelineStage, PipelineStatus

class FailureType(str, Enum):
    MISSING_CORPUS = "MISSING_CORPUS"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    CHUNK_BOUNDARY = "CHUNK_BOUNDARY"
    UNSUPPORTED_GENERATION = "UNSUPPORTED_GENERATION"
    CONTRADICTORY_GENERATION = "CONTRADICTORY_GENERATION"
    GROUNDING_FAILURE = "GROUNDING_FAILURE"
    MULTI_HOP_REASONING_FAILURE = "MULTI_HOP_REASONING_FAILURE"
    UNKNOWN = "UNKNOWN"

@dataclass
class RootCauseAnalysis:
    trace_id: str
    artifact_version: str = "1.0"
    primary_cause: FailureType = FailureType.UNKNOWN
    secondary_effects: List[FailureType] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0
    recommendations_needed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        data = asdict(self)
        data["primary_cause"] = data["primary_cause"].value
        data["secondary_effects"] = [e.value for e in data["secondary_effects"]]
        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, data_str: str) -> 'RootCauseAnalysis':
        data = json.loads(data_str)
        data["primary_cause"] = FailureType(data["primary_cause"])
        data["secondary_effects"] = [FailureType(e) for e in data.get("secondary_effects", [])]
        return cls(**data)

    def save(self, base_dir: str = "artifacts/root_cause_analysis") -> str:
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"TRACE_{self.trace_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        return filepath

    @classmethod
    def load(cls, filepath: str) -> 'RootCauseAnalysis':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())


class RootCauseReasoner:
    
    STAGE_TO_FAILURE_MAP = {
        PipelineStage.CORPUS: FailureType.MISSING_CORPUS,
        PipelineStage.RETRIEVER: FailureType.RETRIEVAL_MISS,
        PipelineStage.CHUNKING: FailureType.CHUNK_BOUNDARY,
        PipelineStage.GENERATOR: FailureType.UNSUPPORTED_GENERATION,
        PipelineStage.GROUNDING: FailureType.GROUNDING_FAILURE
    }

    def analyze(self, psm: PipelineStateMatrix) -> RootCauseAnalysis:
        # Traverse stages in strict causal order
        traversal_order = [
            PipelineStage.CORPUS,
            PipelineStage.RETRIEVER,
            PipelineStage.CHUNKING,
            PipelineStage.GENERATOR,
            PipelineStage.GROUNDING
        ]
        
        primary_cause = None
        secondary_effects = []
        reasoning_chain = []
        confidence_scores = []
        
        reasoning_chain.append(f"Starting root cause analysis for trace {psm.trace_id}.")
        
        for stage in traversal_order:
            state = psm.get(stage)
            if not state:
                reasoning_chain.append(f"Skipping {stage.value}: No observable state found in matrix.")
                continue
                
            if state.status == PipelineStatus.UNKNOWN:
                reasoning_chain.append(f"Skipping {stage.value}: State is UNKNOWN. Insufficient evidence.")
                continue
                
            if state.status == PipelineStatus.FAIL:
                failure_type = self.STAGE_TO_FAILURE_MAP.get(stage, FailureType.UNKNOWN)
                if primary_cause is None:
                    primary_cause = failure_type
                    reasoning_chain.append(f"Primary Cause Identified at {stage.value}: {failure_type.value}. Observation: {state.observation}")
                else:
                    secondary_effects.append(failure_type)
                    reasoning_chain.append(f"Secondary Effect Logged at {stage.value}: {failure_type.value}. Observation: {state.observation}")
                    
                confidence_scores.append(state.confidence)
            elif state.status == PipelineStatus.PASS:
                reasoning_chain.append(f"Assumption Passed at {stage.value}. Moving to next stage.")
                confidence_scores.append(state.confidence)

        final_primary_cause = primary_cause if primary_cause else FailureType.UNKNOWN
        if final_primary_cause == FailureType.UNKNOWN:
            reasoning_chain.append("No failures detected. Pipeline is healthy or evidence is purely UNKNOWN.")
            needs_rec = False
        else:
            reasoning_chain.append(f"Concluding analysis with primary cause: {final_primary_cause.value}.")
            needs_rec = True
            
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        return RootCauseAnalysis(
            trace_id=psm.trace_id,
            primary_cause=final_primary_cause,
            secondary_effects=secondary_effects,
            reasoning_chain=reasoning_chain,
            confidence=round(avg_confidence, 2),
            recommendations_needed=needs_rec
        )
