"""
Test script to run the pipeline up to the Embedding Engine.
"""
import os
import sys
import logging

# Add the project root to the python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.ingestion import load_documents_from_directory
from src.chunk_engine import create_chunks
from src.chunk_registry import ChunkRegistry
from src.embedding_engine import generate_embeddings

logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    data_dir = os.path.join(project_root, "data")
    
    print("-" * 50)
    print("STEP 1: PIPELINE PREPARATION")
    print("-" * 50)
    
    try:
        documents = load_documents_from_directory(data_dir)
        if not documents:
            print("No documents found. Please place a PDF in the 'data' directory.")
            return
        
        chunks = create_chunks(documents)
        
        registry = ChunkRegistry()
        registry.register(chunks)
        
    except Exception as e:
        print(f"Failed during pipeline preparation: {e}")
        return

    print("-" * 50)
    print("STEP 2: GENERATING EMBEDDINGS")
    print("-" * 50)
    
    # Generate embeddings based on the canonical registry
    embeddings = generate_embeddings(registry)
    
    print("-" * 50)
    print("RESULTS")
    print("-" * 50)
    
    print(f"Number of embeddings generated: {len(embeddings)}")
    
    if embeddings:
        first_emb = embeddings[0]
        print("\n--- Example Embedding Output ---")
        print(f"Chunk ID:          {first_emb.chunk_id}")
        print(f"Parent Doc ID:     {first_emb.parent_document_id}")
        print(f"Embedding Model:   {first_emb.embedding_model}")
        print(f"Embedding Dim:     {first_emb.embedding_dimension}")
        print(f"Generated at:      {first_emb.timestamp}")
        print(f"Vector (preview):  {first_emb.embedding[:5]}... (total {len(first_emb.embedding)} floats)")

if __name__ == "__main__":
    run_test()
