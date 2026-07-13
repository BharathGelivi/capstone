import os
import argparse
from src.chunk_registry import ChunkRegistry
from src.vector_store import ChromaVectorStore
from src.retriever import Retriever
from src.generator import Generator

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

if __name__ == "__main__":
    main()
