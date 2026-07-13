"""
Test script to run the Document Loader and Chunk Engine together.
"""
import os
import sys
import logging

# Add the project root to the python path so we can import src modules easily
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.ingestion import load_documents_from_directory
from src.chunk_engine import create_chunks

# Suppress LlamaIndex's noisy info logs for this test script, keep our own
logging.getLogger("llama_index").setLevel(logging.WARNING)

def run_test():
    # 1. Load documents
    data_dir = os.path.join(project_root, "data")
    
    print("-" * 50)
    print("STEP 1: LOADING DOCUMENTS")
    print("-" * 50)
    
    try:
        documents = load_documents_from_directory(data_dir)
    except Exception as e:
        print(f"Failed to load documents: {e}")
        print("Please place at least one PDF file in the 'data' directory and try again.")
        return

    if not documents:
        print("No documents found. Please place a PDF in the 'data' directory.")
        return
        
    print("-" * 50)
    print("STEP 2: CHUNKING DOCUMENTS")
    print("-" * 50)
    
    # 2. Pass them to the chunk engine
    chunks = create_chunks(documents)
    
    # 3. Print the results
    print("-" * 50)
    print("RESULTS")
    print("-" * 50)
    
    print(f"Number of documents (pages): {len(documents)}")
    print(f"Number of chunks created:    {len(chunks)}")
    
    if chunks:
        first_chunk = chunks[0]
        print("\n--- Metadata of the First Chunk ---")
        for key, value in first_chunk.metadata.items():
            print(f"  {key}: {value}")
            
        print("\n--- Content of the First Chunk (First 300 chars) ---")
        print(first_chunk.text[:300] + "...")

if __name__ == "__main__":
    run_test()
