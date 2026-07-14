import os
import json
import uuid
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any
from datetime import datetime, timezone

from src.root_cause_reasoner import RootCauseAnalysis, FailureType

class ActionCategory(str, Enum):
    RETRIEVAL = "RETRIEVAL"
    CHUNKING = "CHUNKING"
    GENERATION = "GENERATION"
    CORPUS = "CORPUS"
    SYSTEM = "SYSTEM"

@dataclass
class CorrectiveAction:
    action_id: str
    category: ActionCategory
    title: str
    description: str
    observed_evidence: str
    root_cause: str
    expected_improvement: str
    success_metric: str
    tradeoff: str
    priority: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CorrectiveActionPlan:
    trace_id: str
    primary_cause: FailureType
    artifact_version: str = "1.0"
    immediate_actions: List[CorrectiveAction] = field(default_factory=list)
    short_term_actions: List[CorrectiveAction] = field(default_factory=list)
    experimental_actions: List[CorrectiveAction] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        data = asdict(self)
        data["primary_cause"] = data["primary_cause"].value
        for action_list in ["immediate_actions", "short_term_actions", "experimental_actions"]:
            for action in data[action_list]:
                action["category"] = action["category"].value
        return json.dumps(data, indent=4)

    @classmethod
    def from_json(cls, data_str: str) -> 'CorrectiveActionPlan':
        data = json.loads(data_str)
        data["primary_cause"] = FailureType(data["primary_cause"])
        
        for action_list in ["immediate_actions", "short_term_actions", "experimental_actions"]:
            actions = []
            for act_data in data.get(action_list, []):
                act_data["category"] = ActionCategory(act_data["category"])
                actions.append(CorrectiveAction(**act_data))
            data[action_list] = actions
            
        return cls(**data)

    def save(self, base_dir: str = "artifacts/corrective_action_plan") -> str:
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, f"TRACE_{self.trace_id}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        return filepath

    @classmethod
    def load(cls, filepath: str) -> 'CorrectiveActionPlan':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

class CorrectiveActionEngine:
    
    def __init__(self):
        self.lookup_table = self._initialize_lookup_table()

    def _initialize_lookup_table(self) -> Dict[FailureType, List[Dict[str, str]]]:
        return {
            FailureType.MISSING_CORPUS: [
                {
                    "category": ActionCategory.CORPUS,
                    "title": "Ingest Missing Domain Knowledge",
                    "description": "Expand corpus to cover entities not found in the vector index.",
                    "observed_evidence": "All retrieved chunks have low similarity scores for given keywords.",
                    "root_cause": "The vector store simply does not contain the required information.",
                    "expected_improvement": "Higher retrieval relevance scores for out-of-domain queries.",
                    "success_metric": "Increase in average chunk similarity score > 0.70.",
                    "tradeoff": "Increases index size and embedding costs.",
                    "priority": "immediate"
                }
            ],
            FailureType.RETRIEVAL_MISS: [
                {
                    "category": ActionCategory.RETRIEVAL,
                    "title": "Increase Retrieval Top-K",
                    "description": "Increase the number of retrieved documents before providing context to the LLM.",
                    "observed_evidence": "Ground truth or highly relevant chunks are present but ranked lower than current Top-K.",
                    "root_cause": "Semantic similarity metric fails to rank the most contextually relevant chunks at the very top.",
                    "expected_improvement": "Higher recall; missing context is caught within the expanded window.",
                    "success_metric": "Verification support score increases above 0.8.",
                    "tradeoff": "Increases prompt token usage, latency, and cost.",
                    "priority": "immediate"
                },
                {
                    "category": ActionCategory.RETRIEVAL,
                    "title": "Implement Hybrid Search (Dense + Sparse)",
                    "description": "Add BM25 lexical search and combine with semantic vector search using Reciprocal Rank Fusion.",
                    "observed_evidence": "Queries containing specific IDs, acronyms, or names fail semantic retrieval.",
                    "root_cause": "Dense embeddings often fail at exact keyword matching.",
                    "expected_improvement": "Near-perfect recall for exact-match terminology.",
                    "success_metric": "Retrieval hit rate for exact entity queries approaches 1.0.",
                    "tradeoff": "Requires maintaining a secondary inverted index and fusion overhead.",
                    "priority": "short_term"
                },
                {
                    "category": ActionCategory.RETRIEVAL,
                    "title": "Deploy Cross-Encoder Reranker",
                    "description": "Pass the Top-N results through a Cross-Encoder to re-score query-document relevance.",
                    "observed_evidence": "Bi-encoder returns top-k chunks that share vocabulary but lack true semantic relevance to the question.",
                    "root_cause": "Bi-encoders calculate dot products independently and miss deep query-document interactions.",
                    "expected_improvement": "Drastically improved precision at Top-1 and Top-3.",
                    "success_metric": "Precision@3 increases by at least 20%.",
                    "tradeoff": "High computational cost; increases retrieval latency by 100-300ms.",
                    "priority": "experimental"
                }
            ],
            FailureType.CHUNK_BOUNDARY: [
                {
                    "category": ActionCategory.CHUNKING,
                    "title": "Increase Chunk Overlap",
                    "description": "Adjust text splitter to increase overlap between adjacent chunks by 10-20%.",
                    "observed_evidence": "LLM answers partially but misses critical context that was split into the next unretrieved chunk.",
                    "root_cause": "Hard boundary splits separated a continuous semantic thought.",
                    "expected_improvement": "Context is preserved across chunks.",
                    "success_metric": "Reduction in 'Partially Supported' claims.",
                    "tradeoff": "Increases total chunk count, redundancy, and storage.",
                    "priority": "immediate"
                }
            ],
            FailureType.UNSUPPORTED_GENERATION: [
                {
                    "category": ActionCategory.GENERATION,
                    "title": "Lower Sampling Temperature",
                    "description": "Reduce generator temperature to 0.0 or 0.1 for deterministic outputs.",
                    "observed_evidence": "LLM generates facts that sound plausible but do not exist in the retrieved text.",
                    "root_cause": "High temperature allows the LLM to creatively sample from its parametric memory rather than strict context.",
                    "expected_improvement": "Output becomes rigid and closely aligned to the prompt context.",
                    "success_metric": "Hallucination rate (Unsupported claims) drops to near 0%.",
                    "tradeoff": "Responses may become dry, repetitive, or overly cautious.",
                    "priority": "immediate"
                },
                {
                    "category": ActionCategory.GENERATION,
                    "title": "Strict Prompt Grounding",
                    "description": "Inject systemic instructions forcing the LLM to output 'I don't know' if context is insufficient.",
                    "observed_evidence": "The LLM guesses when retrieved context is irrelevant.",
                    "root_cause": "The prompt does not strongly penalize guessing.",
                    "expected_improvement": "LLM reliably declines to answer ungrounded queries.",
                    "success_metric": "Increase in 'Not Verifiable' traces rather than 'Unsupported' claims.",
                    "tradeoff": "May refuse to answer queries that could be logically deduced.",
                    "priority": "short_term"
                }
            ],
            FailureType.GROUNDING_FAILURE: [
                {
                    "category": ActionCategory.GENERATION,
                    "title": "Require Explicit Citations",
                    "description": "Force the LLM to append [Chunk-ID] to every sentence it generates.",
                    "observed_evidence": "LLM struggles to maintain factual fidelity across long generation spans.",
                    "root_cause": "Lack of forced intermediate reasoning (citations) causes drift from context.",
                    "expected_improvement": "Self-correction during generation ensures every fact is linked to evidence.",
                    "success_metric": "100% of claims map cleanly to a cited chunk.",
                    "tradeoff": "Increases output token length and adds parsing overhead.",
                    "priority": "experimental"
                }
            ]
        }

    def generate(self, rca: RootCauseAnalysis) -> CorrectiveActionPlan:
        immediate = []
        short_term = []
        experimental = []
        
        failures_to_process = set()
        if rca.primary_cause != FailureType.UNKNOWN:
            failures_to_process.add(rca.primary_cause)
        failures_to_process.update(rca.secondary_effects)

        for failure in failures_to_process:
            templates = self.lookup_table.get(failure, [])
            for template in templates:
                action = CorrectiveAction(
                    action_id=str(uuid.uuid4()),
                    category=template["category"],
                    title=template["title"],
                    description=template["description"],
                    observed_evidence=template["observed_evidence"],
                    root_cause=template["root_cause"],
                    expected_improvement=template["expected_improvement"],
                    success_metric=template["success_metric"],
                    tradeoff=template["tradeoff"],
                    priority=template["priority"]
                )
                
                if action.priority == "immediate":
                    immediate.append(action)
                elif action.priority == "short_term":
                    short_term.append(action)
                else:
                    experimental.append(action)

        return CorrectiveActionPlan(
            trace_id=rca.trace_id,
            primary_cause=rca.primary_cause,
            immediate_actions=immediate,
            short_term_actions=short_term,
            experimental_actions=experimental
        )
