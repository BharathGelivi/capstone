"""
Embedding Engine Module.
Converts text chunks into numerical vector embeddings.
This module strictly isolates embedding generation from storage.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.chunk_registry import ChunkRegistry
from configs.models import EMBEDDING_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EmbeddingRecord:
    """
    A model representing a generated embedding and its critical metadata.
    This maintains the link between the raw chunk and its mathematical representation.
    """
    chunk_id: str
    parent_document_id: str
    embedding: List[float]
    embedding_model: str
    embedding_dimension: int
    timestamp: str

def generate_embeddings(registry: ChunkRegistry) -> List[EmbeddingRecord]:
    """
    Takes a populated ChunkRegistry, generates embeddings for every text chunk,
    and returns a collection of EmbeddingRecords.
    
    Args:
        registry (ChunkRegistry): The registry containing chunk origins and text.
        
    Returns:
        List[EmbeddingRecord]: A list of objects containing the embeddings and metadata.
    """
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    
    # Initialize the HuggingFace embedding model.
    # LlamaIndex will download the model weights (if not already cached) and run it locally.
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
    
    # Optional: fetch a test embedding to determine the dimension dynamically
    sample_emb = embed_model.get_text_embedding("test")
    dimension = len(sample_emb)
    logger.info(f"Model initialized successfully. Embedding dimension: {dimension}")
    
    records_to_process = list(registry._records.values())
    logger.info(f"Generating embeddings for {len(records_to_process)} chunks...")
    
    embedding_records: List[EmbeddingRecord] = []
    
    for i, record in enumerate(records_to_process):
        # We generate the vector representation of the chunk's text
        vector = embed_model.get_text_embedding(record.text)
        
        # Create an EmbeddingRecord linking the new vector back to the chunk
        emb_record = EmbeddingRecord(
            chunk_id=record.chunk_id,
            parent_document_id=record.parent_document_id,
            embedding=vector,
            embedding_model=EMBEDDING_MODEL_NAME,
            embedding_dimension=len(vector),
            timestamp=datetime.utcnow().isoformat()
        )
        
        embedding_records.append(emb_record)
        
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(records_to_process)} chunks.")
            
    logger.info("Embedding generation complete.")
    return embedding_records
