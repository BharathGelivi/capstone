"""
Test script to run the pipeline up to the Vector Store.
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
from src.vector_store import ChromaVectorStore
from src.config import CHROMA_COLLECTION_NAME

logging.getLogger("llama_index").setLevel(logging.WARNING)
# Suppress chromadb telemetry logs
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    data_dir = os.path.join(project_root, "data")
    
    print("-" * 50)
    print("STEP 1: PREPARATION (Loading, Chunking, Embs)")
    print("-" * 50)
    
    try:
        documents = load_documents_from_directory(data_dir)
        if not documents:
            print("No documents found. Please place a PDF in the 'data' directory.")
            return
            
        chunks = create_chunks(documents)
        
        registry = ChunkRegistry()
        registry.register(chunks)
        
        embeddings = generate_embeddings(registry)
        
    except Exception as e:
        print(f"Failed during pipeline preparation: {e}")
        return

    print("-" * 50)
    print("STEP 2: VECTOR STORE OPERATIONS")
    print("-" * 50)
    
    try:
        # Initialize the store
        store = ChromaVectorStore()
        store.initialize_collection()
        
        # We can optionally clear out old data for this test to be clean
        # But we'll just upsert which handles overwrites gracefully.
        
        # Insert embeddings
        store.add_embeddings(embeddings, registry)
        
        print("-" * 50)
        print("RESULTS")
        print("-" * 50)
        
        vector_count = store.count()
        print(f"Collection Name:      {CHROMA_COLLECTION_NAME}")
        print(f"Total Vectors Stored: {vector_count}")
        
        if vector_count > 0:
            first_chunk_id = embeddings[0].chunk_id
            print(f"First Stored ChunkID: {first_chunk_id}")
            
            # Verify retrieval by ID
            retrieved = store.get_by_chunk_id(first_chunk_id)
            if retrieved:
                print("\n--- Example Stored Metadata ---")
                for k, v in retrieved['metadata'].items():
                    print(f"  {k}: {v} ({type(v).__name__})")
                print(f"\n--- Stored Document Text (preview) ---\n  {retrieved['document'][:100]}...")
            else:
                print("Failed to retrieve the inserted chunk!")
                
    except Exception as e:
        print(f"Error in Vector Store operations: {e}")

if __name__ == "__main__":
    run_test()
