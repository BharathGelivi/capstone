import os
import argparse
from dotenv import load_dotenv
from src.chunk_registry import ChunkRegistry
from src.vector_store import ChromaVectorStore
from src.retriever import Retriever
from src.generator import Generator

# Load environment variables from .env file
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Query the RAG benchmark system.")
    parser.add_argument("query", type=str, help="The question to ask the system.")
    args = parser.parse_args()

    # 1. Initialize the Vector Store
    print("--- Initializing Vector Store ---")
    vector_store = ChromaVectorStore()
    vector_store.initialize_collection()

    # 2. Load the Chunk Registry
    print("--- Loading Chunk Registry ---")
    registry_path = "artifacts/chunk_registry.json"
    if not os.path.exists(registry_path):
        print(f"Error: {registry_path} not found. Please run run_pipeline.py first.")
        return
    registry = ChunkRegistry.load_from_json(registry_path)

    # 3. Retrieve chunks
    print(f"\n--- Retrieving context for: '{args.query}' ---")
    retriever = Retriever(vector_store=vector_store, chunk_registry=registry)
    retrieval_result = retriever.retrieve(args.query)

    print(f"Found {len(retrieval_result.retrieved_chunks)} relevant chunks in {retrieval_result.retrieval_time:.2f}s.")
    for i, chunk in enumerate(retrieval_result.retrieved_chunks, start=1):
        print(f"Retrieved Chunk {i}")
        print("-" * 40)
        print(f"\nScore:\n{chunk.similarity_score:.3f}")
        print(f"\nSource:\n{chunk.source_file}")
        print(f"\nPage:\n{chunk.page_number}")
        print(f"\nChunk ID:\n{chunk.chunk_id}")
        print(f"\nText:\n\n{chunk.chunk_text}")
        print("\n" + "-" * 40 + "\n")

    # 4. Generate Answer
    # NOTE: The Generator requires the HF_TOKEN environment variable to be set.
    print("\n--- Generating Answer ---")
    if not os.environ.get("HF_TOKEN"):
        print("\n[WARNING] HF_TOKEN environment variable is not set!")
        print("Please set it in your terminal before running this script.")
        print("Example (Windows): $env:HF_TOKEN='your_api_key'")
        return

    generator = Generator()
    generation_result = generator.generate(retrieval_result)

    print(f"\nAnswer generated in {generation_result.generation_time:.2f}s:")
    print("=" * 60)
    print(generation_result.generated_answer)
    print("=" * 60)

    # 5. Build RAGTrace
    print("\n--- Building RAGTrace ---")
    from src.rag_trace import RAGTraceBuilder
    total_time = retrieval_result.retrieval_time + generation_result.generation_time
    trace = RAGTraceBuilder.build(retrieval_result, generation_result, total_time)
    trace_path = RAGTraceBuilder.save_to_json(trace)
    print(f"RAGTrace saved to {trace_path}")

    # 6. Decompose Claims
    print("\n--- Decomposing Answer into Claims ---")
    from src.claim_decomposer import ClaimDecomposer
    decomposer = ClaimDecomposer()
    candidate_claim_set = decomposer.decompose(trace)
    print(f"Extracted {candidate_claim_set.total_candidates} claims.")

    # Convert CandidateClaimSet to canonical ClaimSet
    from src.claims import ClaimSet, ClaimFactory
    claim_factory = ClaimFactory(trace.trace_id)
    claim_set = ClaimSet(trace_id=trace.trace_id)
    for c in candidate_claim_set.candidate_claims:
        claim = claim_factory.create_claim(
            claim_text=c.claim_text,
            source_sentence=c.source_sentence,
            sentence_id=c.sentence_id,
            character_start=c.character_start,
            character_end=c.character_end
        )
        claim_set.add_claim(claim)
    claim_set.to_json(f"artifacts/claims/TRACE_{trace.trace_id}.json")

    # 7. Verify Claims
    print("\n--- Verifying Claims against Evidence ---")
    from src.claim_verifier import ClaimVerifier
    verifier = ClaimVerifier()
    verification_summary = verifier.verify_all(candidate_claim_set, trace.trace_id, retrieval_result.retrieved_chunks)
    verifier.save_artifacts(verification_summary)
    print(f"Verification complete. Supported: {verification_summary.supported_claims}/{verification_summary.total_claims}")

    # 8. Analyze Pipeline State
    print("\n--- Generating Pipeline State Matrix ---")
    from src.pipeline_state_analyzer import PipelineStateAnalyzer
    analyzer = PipelineStateAnalyzer()
    psm = analyzer.analyze(trace, claim_set, verification_summary)
    psm_path = psm.save()
    print(f"Pipeline State Matrix saved to: {psm_path}")
    print("\nFinal Pipeline Summary:")
    for stage, status in psm.summary().items():
        print(f"  {stage}: {status}")

    # 9. Root Cause Reasoner
    print("\n--- Running Root Cause Analysis ---")
    from src.root_cause_reasoner import RootCauseReasoner
    reasoner = RootCauseReasoner()
    rca = reasoner.analyze(psm)
    rca_path = rca.save()
    print(f"Root Cause Analysis saved to: {rca_path}")
    print(f"Primary Cause: {rca.primary_cause.value}")
    if rca.secondary_effects:
        print(f"Secondary Effects: {[e.value for e in rca.secondary_effects]}")
    
    # 10. Corrective Action Engine
    print("\n--- Generating Corrective Action Plan ---")
    from src.corrective_action_engine import CorrectiveActionEngine
    cae = CorrectiveActionEngine()
    plan = cae.generate(rca)
    plan_path = plan.save()
    print(f"Corrective Action Plan saved to: {plan_path}")
    total_actions = len(plan.immediate_actions) + len(plan.short_term_actions) + len(plan.experimental_actions)
    if total_actions > 0:
        print(f"Found {total_actions} actionable corrective measures. Check the plan for details!")
    else:
        print("No corrective actions needed. Pipeline is healthy!")

if __name__ == "__main__":
    main()
