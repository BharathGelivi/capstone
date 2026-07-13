"""
Test script for the Retriever module.
"""
import os
import sys
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.ingestion import load_documents_from_directory
from src.chunk_engine import create_chunks
from src.chunk_registry import ChunkRegistry
from src.embedding_engine import generate_embeddings
from src.vector_store import ChromaVectorStore
from src.retriever import Retriever

logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    data_dir = os.path.join(project_root, "data")
    
    print("-" * 50)
    print("STEP 1: FULL PIPELINE INITIALIZATION")
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
        
        store = ChromaVectorStore()
        store.initialize_collection()
        store.add_embeddings(embeddings, registry)
        
    except Exception as e:
        print(f"Failed during pipeline initialization: {e}")
        return

    print("-" * 50)
    print("STEP 2: RETRIEVAL TESTING")
    print("-" * 50)
    
    retriever = Retriever(vector_store=store, chunk_registry=registry)
    
    # Test Questions
    questions = [
        "What is the main objective of this document?",
        "Explain the key methodology used."
    ]
    
    for q in questions:
        print(f"\n--- Question: '{q}' ---")
        result = retriever.retrieve(q)
        
        print(f"Retrieval Time: {result.retrieval_time:.4f} seconds")
        print(f"Top-{result.top_k} Retrieved Chunks:")
        
        for chunk in result.retrieved_chunks:
            print(f"\n  [Rank {chunk.rank}] Score/Distance: {chunk.similarity_score:.4f} | Chunk ID: {chunk.chunk_id}")
            print(f"  Source: {chunk.source_file} | Page: {chunk.page_number}")
            print(f"  Snippet: {chunk.chunk_text[:100]}...")

if __name__ == "__main__":
    run_test()
