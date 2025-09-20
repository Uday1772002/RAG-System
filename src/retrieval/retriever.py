"""
Retrieval Module
Handles document search and retrieval for RAG systems
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Fix relative imports
try:
    from ..embeddings.embedding_generator import EmbeddingGenerator, VectorStore
except ImportError:
    from src.embeddings.embedding_generator import EmbeddingGenerator, VectorStore

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Represents a retrieval result with metadata"""
    text: str
    page_number: int
    chunk_id: str
    similarity_score: float
    metadata: Dict
    source_document: str

class DocumentRetriever:
    """Retrieves relevant documents for given queries"""
    
    def __init__(self, vector_store, embedding_generator):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        
    def retrieve(self, query: str, top_k: int = 5, 
                 filter_metadata: Optional[Dict] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of retrieval results
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_single_embedding(query)
            
            # Search vector store
            search_results = self.vector_store.search(
                query_embedding, 
                top_k=top_k,
                filter_metadata=filter_metadata
            )
            
            # Convert to RetrievalResult objects
            retrieval_results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    text=result['text'],
                    page_number=result['metadata'].get('page', 0),
                    chunk_id=result['id'],
                    similarity_score=1 - result['distance'],  # Convert distance to similarity
                    metadata=result['metadata'],
                    source_document=result['metadata'].get('file_path', 'unknown')
                )
                retrieval_results.append(retrieval_result)
                
            # Sort by similarity score
            retrieval_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            logger.info(f"Retrieved {len(retrieval_results)} documents for query: {query}")
            return retrieval_results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise
            
    def retrieve_with_reranking(self, query: str, top_k: int = 5,
                               filter_metadata: Optional[Dict] = None) -> List[RetrievalResult]:
        """
        Retrieve documents with additional reranking
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of reranked retrieval results
        """
        # First, retrieve more candidates
        candidates = self.retrieve(query, top_k * 2, filter_metadata)
        
        if not candidates:
            return []
            
        # Apply additional reranking logic
        reranked_results = self._rerank_results(query, candidates)
        
        # Return top_k results
        return reranked_results[:top_k]
        
    def _rerank_results(self, query: str, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Apply additional reranking to candidates"""
        # Simple keyword-based reranking
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for candidate in candidates:
            # Calculate keyword overlap score
            candidate_lower = candidate.text.lower()
            candidate_words = set(candidate_lower.split())
            
            # Jaccard similarity for keyword overlap
            intersection = len(query_words.intersection(candidate_words))
            union = len(query_words.union(candidate_words))
            
            if union > 0:
                keyword_score = intersection / union
                # Combine with similarity score (70% similarity, 30% keyword)
                candidate.similarity_score = (
                    0.7 * candidate.similarity_score + 
                    0.3 * keyword_score
                )
                
        # Sort by combined score
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        return candidates
        
    def get_document_context(self, chunk_id: str, context_window: int = 2) -> Dict:
        """
        Get document context around a specific chunk
        
        Args:
            chunk_id: ID of the target chunk
            context_window: Number of chunks to include on each side
            
        Returns:
            Dictionary with target chunk and context
        """
        try:
            # This would require additional implementation to get neighboring chunks
            # For now, return basic chunk information
            return {
                "target_chunk": chunk_id,
                "context_window": context_window,
                "note": "Context retrieval not yet implemented"
            }
        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            return {}
            
    def search_by_metadata(self, metadata_filter: Dict) -> List[RetrievalResult]:
        """
        Search documents by metadata criteria
        
        Args:
            metadata_filter: Dictionary of metadata filters
            
        Returns:
            List of matching documents
        """
        try:
            # This would require implementing metadata-based search in the vector store
            # For now, return empty list
            logger.warning("Metadata-based search not yet implemented")
            return []
        except Exception as e:
            logger.error(f"Error in metadata search: {e}")
            return []
