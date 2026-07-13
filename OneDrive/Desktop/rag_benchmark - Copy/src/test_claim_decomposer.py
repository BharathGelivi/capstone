import os
import uuid
from datetime import datetime
import json
import logging
from dotenv import load_dotenv

load_dotenv()

from src.rag_trace import RAGTrace
from src.claim_decomposer import ClaimDecomposer

# Set debug mode for the test
os.environ["CLAIM_DECOMPOSER_DEBUG"] = "True"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_mock_trace(question: str, answer: str) -> RAGTrace:
    return RAGTrace(
        trace_id=str(uuid.uuid4()),
        trace_version="1.0",
        pipeline_version="1.0",
        framework_version="1.0",
        timestamp=datetime.utcnow().isoformat() + "Z",
        question=question,
        generated_answer=answer,
        prompt_snapshot="Mock prompt",
        prompt_length=100,
        retrieved_chunk_references=[],
        configuration_snapshot={},
        execution_statistics={},
        pipeline_stage_status={},
        diagnostics=None
    )

def test_claim_decomposer():
    # Use Qwen2.5-7B-Instruct because Hugging Face Serverless API supports it for free accounts
    decomposer = ClaimDecomposer(model_name="Qwen/Qwen2.5-7B-Instruct")
    
    test_cases = [
        {
            "question": "What are the penalties for insider trading?",
            "answer": "Under the Securities Exchange Act, insider trading is punishable by up to 20 years in prison. Additionally, individuals may face fines of up to $5 million, while corporations can be fined up to $25 million."
        },
        {
            "question": "Explain the concept of adverse possession.",
            "answer": "Adverse possession allows a person to claim ownership of land under certain conditions. The possession must be continuous, hostile, open, notorious, and exclusive for a statutory period, which is typically 10 to 20 years depending on the state."
        },
        {
            "question": "Who is liable in a strict liability tort?",
            "answer": "In strict liability, the defendant is liable for damages regardless of intent or negligence. This commonly applies to abnormally dangerous activities, such as keeping wild animals or storing explosives."
        },
        {
            "question": "What is the statute of frauds?",
            "answer": "The statute of frauds requires certain types of contracts to be in writing to be enforceable. These include contracts for the sale of real estate, agreements that cannot be performed within one year, and contracts for the sale of goods over $500."
        },
        {
            "question": "Can a minor enter into a legally binding contract?",
            "answer": "Generally, a contract with a minor is voidable at the minor's discretion. However, contracts for necessities, such as food, clothing, and shelter, are often binding to ensure minors can obtain essential goods and services."
        }
    ]

    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'='*50}\nTest Case {i}\n{'='*50}")
        trace = create_mock_trace(tc["question"], tc["answer"])
        
        print(f"Question: {trace.question}")
        print(f"Generated Answer: {trace.generated_answer}\n")
        
        claim_set = decomposer.decompose(trace)
        
        print(f"Total Claims: {claim_set.total_candidates}")
        
        # Output detailed diagnostics if they exist
        if "diagnostics" in claim_set.metadata:
            print("\nDiagnostics:")
            print(json.dumps(claim_set.metadata["diagnostics"], indent=2))
        else:
            print(f"Metadata: {json.dumps(claim_set.metadata, indent=2)}")
            
        print("\nCandidate Claims:")
        for claim in claim_set.candidate_claims:
            print(f"- [Claim {claim.claim_index}] {claim.claim_text}")
            print(f"  Sentence ID: {claim.sentence_id}")
            print(f"  Position: {claim.character_start} to {claim.character_end}")
            print()

if __name__ == "__main__":
    test_claim_decomposer()
