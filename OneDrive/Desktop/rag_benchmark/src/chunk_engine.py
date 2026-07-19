"""
Chunk Engine Module for RAG Diagnostic Benchmark.
This module handles taking full Document objects and splitting them into smaller,
manageable Node objects (chunks) while preserving rich metadata for diagnostic analysis.
"""

import logging
from typing import List
from llama_index.core.schema import Document, BaseNode, TextNode
from llama_index.core.node_parser import SentenceSplitter

# Import configuration instead of hardcoding
from configs.pipeline import CHUNK_SIZE, CHUNK_OVERLAP

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_chunks(documents: List[Document], 
                  chunk_size: int = CHUNK_SIZE, 
                  chunk_overlap: int = CHUNK_OVERLAP) -> List[BaseNode]:
    """
    Takes a list of LlamaIndex Document objects and splits them into smaller chunks (Nodes).
    It rigorously preserves metadata needed for future diagnostic analysis.
    
    Args:
        documents (List[Document]): The list of full documents to split.
        chunk_size (int): The maximum size of each chunk.
        chunk_overlap (int): The overlap between chunks.
        
    Returns:
        List[BaseNode]: A list of generated node (chunk) objects with rich metadata.
    """
    logger.info(f"Starting chunking process for {len(documents)} documents.")
    logger.info(f"Configuration: Chunk Size = {chunk_size}, Chunk Overlap = {chunk_overlap}")
    
    # SentenceSplitter respects sentence boundaries, preventing words/sentences 
    # from being awkwardly cut in half, which is crucial for generation quality.
    splitter = SentenceSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    all_nodes: List[BaseNode] = []
    
    for doc in documents:
        # LlamaIndex's get_nodes_from_documents splits the text and inherits metadata from the Document.
        # It also automatically calculates start_char_idx and end_char_idx.
        doc_nodes = splitter.get_nodes_from_documents([doc])
        
        # We loop through the generated nodes for this specific document to inject 
        # our custom tracking metadata for the diagnostic framework.
        for index, node in enumerate(doc_nodes):
            # We ensure we are working with TextNodes which carry text and character indices
            if isinstance(node, TextNode):
                # 1. chunk_id: LlamaIndex generates a UUID for node.id_. We keep it as it is stable and unique.
                # 2. parent_document_id: Available via node.ref_doc_id
                # 3 & 4. source_file and page_number: Automatically inherited from the parent Document's metadata
                
                # Injecting our custom diagnostic metadata
                node.metadata["chunk_index"] = index
                node.metadata["chunk_size_config"] = chunk_size
                node.metadata["chunk_overlap_config"] = chunk_overlap
                node.metadata["character_start"] = node.start_char_idx
                node.metadata["character_end"] = node.end_char_idx
                
                # Explicitly add parent doc ID just to ensure it's in the metadata dict for easy reading
                node.metadata["parent_document_id"] = node.ref_doc_id
                
                # LlamaIndex includes inherited metadata like 'file_path' and 'page_label'
                # We can standardize their names for our diagnostic framework if desired
                if "file_path" in node.metadata:
                    node.metadata["source_file"] = node.metadata["file_path"]
                if "page_label" in node.metadata:
                    node.metadata["page_number"] = node.metadata["page_label"]
                    
            all_nodes.append(node)
            
    logger.info(f"Chunking complete. Generated {len(all_nodes)} total chunks.")
    return all_nodes
