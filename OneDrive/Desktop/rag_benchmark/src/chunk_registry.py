"""
Chunk Registry Module.
Provides a canonical, independent registry for tracking every chunk produced during ingestion.
Essential for the RAG diagnostic framework to map evidence, attribute failures, and report errors.
"""

import os
import json
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from llama_index.core.schema import BaseNode

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """
    A lightweight, serializable model representing a single chunk of text.
    It serves as the ground truth for chunk origin and metadata within the diagnostic framework.
    """
    chunk_id: str
    parent_document_id: str
    source_file: str
    page_number: str
    chunk_index: int
    configured_chunk_size: int
    configured_chunk_overlap: int
    character_start: int
    character_end: int
    text: str
    text_length: int
    metadata: Dict


class ChunkRegistry:
    """
    Maintains a canonical registry of every chunk produced during ingestion.
    Provides methods to query chunks by ID or parent document and computes corpus statistics.
    """
    def __init__(self):
        # We use a dictionary keyed by chunk_id for O(1) fast lookups.
        self._records: Dict[str, ChunkRecord] = {}

    def register(self, nodes: List[BaseNode]):
        """
        Parses LlamaIndex nodes and registers them as ChunkRecords.
        
        Args:
            nodes (List[BaseNode]): The nodes produced by the Chunk Engine.
        """
        logger.info(f"Registering {len(nodes)} chunks into the registry.")
        
        for node in nodes:
            # We extract properties safely. If a metadata field is missing, we provide a default.
            record = ChunkRecord(
                chunk_id=node.id_,
                parent_document_id=node.ref_doc_id or "unknown",
                source_file=node.metadata.get("source_file", "unknown"),
                page_number=str(node.metadata.get("page_number", "unknown")),
                chunk_index=node.metadata.get("chunk_index", -1),
                configured_chunk_size=node.metadata.get("chunk_size_config", -1),
                configured_chunk_overlap=node.metadata.get("chunk_overlap_config", -1),
                character_start=node.metadata.get("character_start", -1),
                character_end=node.metadata.get("character_end", -1),
                text=node.text,
                text_length=len(node.text),
                metadata=node.metadata.copy()  # Save a copy of all other metadata just in case
            )
            self._records[record.chunk_id] = record
            
        logger.info(f"Registry now contains {len(self._records)} records.")

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        """Retrieves a single chunk by its ID."""
        return self._records.get(chunk_id)

    def get_document_chunks(self, document_id: str) -> List[ChunkRecord]:
        """Retrieves all chunks that belong to a specific document."""
        return [record for record in self._records.values() if record.parent_document_id == document_id]

    def total_chunks(self) -> int:
        """Returns the total number of registered chunks."""
        return len(self._records)

    def get_statistics(self) -> Dict[str, float]:
        """
        Calculates simple statistics over the entire chunk registry.
        Useful for evaluating whether the chunking strategy behaves as expected.
        """
        if not self._records:
            return {
                "num_documents": 0,
                "num_chunks": 0,
                "avg_chunk_length": 0.0,
                "max_chunk_length": 0,
                "min_chunk_length": 0
            }

        unique_docs = set(record.parent_document_id for record in self._records.values())
        lengths = [record.text_length for record in self._records.values()]

        return {
            "num_documents": len(unique_docs),
            "num_chunks": len(self._records),
            "avg_chunk_length": round(sum(lengths) / len(lengths), 2),
            "max_chunk_length": max(lengths),
            "min_chunk_length": min(lengths)
        }

    def save_to_json(self, file_path: str):
        """
        Serializes the registry and saves it to a JSON file.
        
        Args:
            file_path (str): The local path where the JSON file will be saved.
        """
        logger.info(f"Saving registry to {file_path}")
        
        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Convert dataclasses to dicts for JSON serialization
        data_to_save = {chunk_id: asdict(record) for chunk_id, record in self._records.items()}
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=4)
            
        logger.info("Registry saved successfully.")

    @classmethod
    def load_from_json(cls, file_path: str) -> 'ChunkRegistry':
        """
        Loads a serialized registry from a JSON file.
        
        Args:
            file_path (str): The local path to the JSON file.
            
        Returns:
            ChunkRegistry: An initialized registry populated with the loaded records.
        """
        logger.info(f"Loading registry from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Registry file not found: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            data_loaded = json.load(f)
            
        registry = cls()
        
        for chunk_id, record_dict in data_loaded.items():
            # Reconstruct the ChunkRecord dataclass from the dictionary
            record = ChunkRecord(**record_dict)
            registry._records[chunk_id] = record
            
        logger.info(f"Successfully loaded {len(registry._records)} records.")
        return registry
