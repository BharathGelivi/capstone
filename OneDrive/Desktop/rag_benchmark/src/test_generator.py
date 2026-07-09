"""
Test script for the Generator module.
Runs the entire pipeline from document loading to final generation.
"""
import os
import sys
import logging
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Load environment variables (crucial for GEMINI_API_KEY)
load_dotenv(os.path.join(project_root, ".env"))

from src.ingestion import load_documents_from_directory
from src.chunk_engine import create_chunks
from src.chunk_registry import ChunkRegistry
from src.embedding_engine import generate_embeddings
from src.vector_store import ChromaVectorStore
from src.retriever import Retriever
from src.generator import Generator

logging.getLogger("llama_index").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_test():
    if not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY environment variable is not set!")
        print("Please create a .env file in the project root with your API key.")
        return
        
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
    print("STEP 2: RETRIEVAL & GENERATION TESTING")
    print("-" * 50)
    
    retriever = Retriever(vector_store=store, chunk_registry=registry)
    generator = Generator()
    
    questions = [
        "What is the main objective of this document?",
        "Explain the key methodology used."
    ]
    
    for q in questions:
        print(f"\n--- Question: '{q}' ---")
        
        # 1. Retrieve
        retrieval_result = retriever.retrieve(q)
        
        # 2. Generate
        generation_result = generator.generate(retrieval_result)
        
        # 3. Output
        print(f"Retrieved Chunk IDs: {generation_result.retrieved_chunk_ids}")
        print(f"Prompt Length:       {generation_result.prompt_length} characters")
        print(f"Model Used:          {generation_result.model_name}")
        print(f"Generation Time:     {generation_result.generation_time:.4f} seconds")
        print(f"\n--- Generated Answer ---")
        print(generation_result.generated_answer)

if __name__ == "__main__":
    run_test()
