import os
import time
import json
import logging
from src.chunk_registry import ChunkRegistry
from src.vector_store import ChromaVectorStore
from src.retriever import Retriever
from src.generator import Generator
from src.rag_trace import RAGTraceBuilder, RAGTrace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test():
    """
    Runs the complete benchmark pipeline, builds a RAGTrace, saves it to JSON,
    reloads it, and prints a summary.
    """
    logger.info("=== Starting RAGTrace End-to-End Test ===")
    
    # Check HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        logger.warning("HF_TOKEN is not set. The generation phase may fail or be rate limited.")
        logger.warning("Please export HF_TOKEN before running this test for real generation.")
    
    start_time = time.time()
    
    # 1. Initialize Vector Store
    logger.info("Initializing Vector Store...")
    vector_store = ChromaVectorStore()
    vector_store.initialize_collection()
    
    # 2. Load Chunk Registry
    registry_path = "artifacts/chunk_registry.json"
    if not os.path.exists(registry_path):
        logger.error(f"{registry_path} not found. Run run_pipeline.py first.")
        return
    registry = ChunkRegistry.load_from_json(registry_path)
    
    # 3. Retrieve
    test_question = "What is the punishment for murder?"
    logger.info(f"Retrieving for question: '{test_question}'")
    retriever = Retriever(vector_store=vector_store, chunk_registry=registry)
    retrieval_result = retriever.retrieve(test_question)
    
    # 4. Generate
    logger.info("Generating answer...")
    generator = Generator()
    generation_result = generator.generate(retrieval_result)
    
    total_pipeline_time = time.time() - start_time
    
    # 5. Build RAGTrace
    logger.info("Building RAGTrace...")
    trace = RAGTraceBuilder.build(
        retrieval_result=retrieval_result, 
        generation_result=generation_result, 
        total_pipeline_time=total_pipeline_time
    )
    
    # 6. Save to JSON
    saved_path = RAGTraceBuilder.save_to_json(trace)
    
    # 7. Reload and Print Summary
    logger.info(f"Reloading trace from {saved_path}...")
    with open(saved_path, 'r', encoding='utf-8') as f:
        reloaded_data = json.load(f)
        
    logger.info("\n" + "="*40 + "\n--- RAGTrace Summary ---\n" + "="*40)
    print(f"Question: {reloaded_data['question']}")
    print(f"Answer Length: {len(reloaded_data['generated_answer'])} characters")
    print(f"Number of Retrieved Chunks: {len(reloaded_data['retrieved_chunk_references'])}")
    print(f"Retrieval Time: {reloaded_data['execution_statistics']['retrieval_time']:.2f}s")
    print(f"Generation Time: {reloaded_data['execution_statistics']['generation_time']:.2f}s")
    print(f"Total Pipeline Time: {reloaded_data['execution_statistics']['total_pipeline_time']:.2f}s")
    print("="*40)
    
if __name__ == "__main__":
    run_test()
