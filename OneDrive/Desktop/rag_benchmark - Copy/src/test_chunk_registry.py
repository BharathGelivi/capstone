"""
Test script to run the Document Loader, Chunk Engine, and Chunk Registry together.
"""
import os
import sys
import logging

# Add the project root to the python path so we can import src modules easily
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.ingestion import load_documents_from_directory
from src.chunk_engine import create_chunks
from src.chunk_registry import ChunkRegistry

# Suppress LlamaIndex's noisy info logs for this test script, keep our own
logging.getLogger("llama_index").setLevel(logging.WARNING)
# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    # 2. Create chunks
    chunks = create_chunks(documents)
    
    print("-" * 50)
    print("STEP 3: BUILDING AND SAVING REGISTRY")
    print("-" * 50)
    
    # 3. Build the Chunk Registry
    registry = ChunkRegistry()
    registry.register(chunks)
    
    # 4. Save the registry
    artifacts_dir = os.path.join(project_root, "artifacts")
    registry_path = os.path.join(artifacts_dir, "chunk_registry.json")
    registry.save_to_json(registry_path)
    
    print("-" * 50)
    print("STEP 4: RELOADING AND VERIFYING REGISTRY")
    print("-" * 50)
    
    # 5. Reload the registry
    reloaded_registry = ChunkRegistry.load_from_json(registry_path)
    
    # 6. Print Results
    stats = reloaded_registry.get_statistics()
    print("\n--- Registry Statistics ---")
    print(f"Total Documents: {stats['num_documents']}")
    print(f"Total Chunks:    {stats['num_chunks']}")
    print(f"Avg Chunk Len:   {stats['avg_chunk_length']} characters")
    print(f"Max Chunk Len:   {stats['max_chunk_length']} characters")
    print(f"Min Chunk Len:   {stats['min_chunk_length']} characters")
    
    if reloaded_registry.total_chunks() > 0:
        # Get an arbitrary chunk from the dictionary
        first_chunk_id = list(reloaded_registry._records.keys())[0]
        first_record = reloaded_registry.get_chunk(first_chunk_id)
        
        print("\n--- First ChunkRecord Attributes ---")
        print(f"Chunk ID:          {first_record.chunk_id}")
        print(f"Parent Doc ID:     {first_record.parent_document_id}")
        print(f"Source File:       {first_record.source_file}")
        print(f"Page Number:       {first_record.page_number}")
        print(f"Chunk Index:       {first_record.chunk_index}")
        print(f"Character Start:   {first_record.character_start}")
        print(f"Text Length:       {first_record.text_length}")
        print(f"Content Snippet:   {first_record.text[:100]}...")

if __name__ == "__main__":
    run_test()
