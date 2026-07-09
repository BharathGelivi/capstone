"""
Retriever Module.
Responsible for converting a user question into an embedding, 
searching the Vector Store (Dense), running BM25 (Sparse), 
and fusing the results using Reciprocal Rank Fusion (RRF).
"""

import time
import logging
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rank_bm25 import BM25Okapi

from src.config import EMBEDDING_MODEL_NAME, RETRIEVAL_TOP_K
from src.vector_store import VectorStore
from src.chunk_registry import ChunkRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RetrievedChunk:
    """
    Represents a single chunk returned from the Hybrid Search.
    """
    chunk_id: str
    similarity_score: float # The final RRF score
    rank: int
    page_number: str
    source_file: str
    chunk_index: int
    chunk_text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    dense_rank: int = -1
    sparse_rank: int = -1

@dataclass
class RetrievalResult:
    """
    A comprehensive result object capturing the entire retrieval event.
    Provides complete transparency for future diagnostics.
    """
    question: str
    question_embedding_dimension: int
    retrieved_chunks: List[RetrievedChunk]
    retrieved_chunk_ids: List[str]
    similarity_scores: List[float]
    retrieval_time: float
    top_k: int
    retrieval_metadata: Dict[str, Any]

class Retriever:
    """
    Executes hybrid search (Dense + Sparse) and fuses results via RRF.
    """
    def __init__(self, vector_store: VectorStore, chunk_registry: ChunkRegistry, top_k: int = RETRIEVAL_TOP_K):
        self.vector_store = vector_store
        self.chunk_registry = chunk_registry
        self.top_k = top_k
        self.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        
        logger.info("Building BM25 Index from ChunkRegistry...")
        self.registry_chunks = list(self.chunk_registry._records.values())
        
        # BM25 Sparse Index Setup
        # We use a regex tokenizer and strip standard English stopwords to prevent
        # common words like 'what', 'does', 'in' from dominating the TF-IDF scores.
        stopwords = {
            "what", "does", "do", "is", "a", "an", "the", "in", "on", "at", "to", "for", 
            "of", "and", "or", "talk", "about", "with", "by", "as", "it", "this", "that",
            "are", "was", "were", "be", "has", "have", "had", "not", "how", "why", "who"
        }
        
        def tokenize(text: str) -> List[str]:
            tokens = re.findall(r'\w+', text.lower())
            return [t for t in tokens if t not in stopwords]
        
        self.bm25_tokenize = tokenize
            
        tokenized_corpus = []
        for record in self.registry_chunks:
            tokenized_corpus.append(self.bm25_tokenize(record.text))
            
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 Index built with {len(self.registry_chunks)} documents.")

    def retrieve(self, question: str) -> RetrievalResult:
        """
        Processes a question using Hybrid Search and returns the most relevant chunks.
        """
        logger.info(f"Initiating Hybrid Retrieval for question: '{question}'")
        start_time = time.time()
        
        # 1. Embed the user question for Dense Retrieval
        question_embedding = self.embed_model.get_text_embedding(question)
        question_dim = len(question_embedding)
        
        # 2a. Dense Search (fetch top 20 for fusion)
        dense_results = self.vector_store.search(query_embedding=question_embedding, top_k=20)
        
        dense_ranks = {}
        dense_scores = {}
        if dense_results and dense_results.get('ids') and len(dense_results['ids'][0]) > 0:
            ids = dense_results['ids'][0]
            distances = dense_results.get('distances', [[0] * len(ids)])[0]
            for rank, (chunk_id, distance) in enumerate(zip(ids, distances), start=1):
                dense_ranks[chunk_id] = rank
                dense_scores[chunk_id] = distance
                
        # 2b. Sparse Search (BM25)
        tokenized_query = self.bm25_tokenize(question)
        bm25_scores_list = self.bm25.get_scores(tokenized_query)
        
        # Sort chunks by BM25 score
        sparse_ranking = sorted(
            zip([record.chunk_id for record in self.registry_chunks], bm25_scores_list),
            key=lambda x: x[1], reverse=True
        )
        
        sparse_ranks = {}
        sparse_scores = {}
        for rank, (chunk_id, score) in enumerate(sparse_ranking[:20], start=1):
            if score > 0: # Only rank if it actually matched keywords
                sparse_ranks[chunk_id] = rank
                sparse_scores[chunk_id] = score
                
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k = 60 # Standard RRF constant
        
        all_candidate_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        
        for chunk_id in all_candidate_ids:
            score = 0.0
            if chunk_id in dense_ranks:
                score += 1.0 / (k + dense_ranks[chunk_id])
            if chunk_id in sparse_ranks:
                score += 1.0 / (k + sparse_ranks[chunk_id])
            rrf_scores[chunk_id] = score
            
        # Sort by final RRF score
        final_ranking = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        
        retrieved_chunks = []
        retrieved_chunk_ids = []
        similarity_scores = []
        
        for final_rank, (chunk_id, rrf_score) in enumerate(final_ranking, start=1):
            registry_record = self.chunk_registry.get_chunk(chunk_id)
            if not registry_record:
                continue
                
            retrieved_chunk = RetrievedChunk(
                chunk_id=chunk_id,
                similarity_score=rrf_score,
                rank=final_rank,
                page_number=str(registry_record.page_number),
                source_file=registry_record.source_file,
                chunk_index=registry_record.chunk_index,
                chunk_text=registry_record.text,
                dense_score=dense_scores.get(chunk_id, 0.0),
                sparse_score=sparse_scores.get(chunk_id, 0.0),
                dense_rank=dense_ranks.get(chunk_id, -1),
                sparse_rank=sparse_ranks.get(chunk_id, -1)
            )
            
            retrieved_chunks.append(retrieved_chunk)
            retrieved_chunk_ids.append(chunk_id)
            similarity_scores.append(rrf_score)
            
        retrieval_time = time.time() - start_time
        
        # 4. Construct RetrievalResult
        result = RetrievalResult(
            question=question,
            question_embedding_dimension=question_dim,
            retrieved_chunks=retrieved_chunks,
            retrieved_chunk_ids=retrieved_chunk_ids,
            similarity_scores=similarity_scores,
            retrieval_time=retrieval_time,
            top_k=self.top_k,
            retrieval_metadata={
                "embedding_model": EMBEDDING_MODEL_NAME,
                "vector_store_type": type(self.vector_store).__name__,
                "retrieval_type": "hybrid_rrf"
            }
        )
        
        logger.info(f"Hybrid Retrieval complete in {retrieval_time:.3f}s. Found {len(retrieved_chunks)} chunks.")
        return result
