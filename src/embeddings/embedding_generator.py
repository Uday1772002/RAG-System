"""
Embedding Generation Module
Creates vector representations of text chunks
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates embeddings for text chunks"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
            
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts with memory optimization
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        try:
            if not texts:
                return np.array([])
                
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Process in smaller batches to reduce memory usage
            batch_size = 8  # Smaller batch size
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(
                    batch_texts, 
                    show_progress_bar=False,  # Disable progress bar to reduce overhead
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normalize to reduce memory
                )
                all_embeddings.append(batch_embeddings)
                
                # Force garbage collection for large batches
                if len(all_embeddings) % 10 == 0:
                    import gc
                    gc.collect()
            
            embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
            logger.info("Embeddings generated successfully")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
            
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.generate_embeddings([text])[0]

class VectorStore:
    """Manages vector storage and retrieval using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./data/vector_db"):
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_db()
        
    def _initialize_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            logger.info("Initializing ChromaDB")
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
            
    def add_documents(self, chunks: List[Dict], embeddings: np.ndarray) -> List[str]:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Numpy array of embeddings
            
        Returns:
            List of document IDs
        """
        try:
            if len(chunks) != len(embeddings):
                raise ValueError("Number of chunks must match number of embeddings")
                
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{chunk['chunk_id']}_{chunk['page_number']}"
                ids.append(chunk_id)
                texts.append(chunk['text'])
                metadatas.append(chunk['metadata'])
                
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
            
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of search results with metadata
        """
        try:
            # Prepare query
            query_embedding_list = [query_embedding.tolist()]
            
            # Perform search
            results = self.collection.query(
                query_embeddings=query_embedding_list,
                n_results=top_k,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(result)
                
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
            
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": self.collection.name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
            
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection("documents")
            logger.info("Collection deleted successfully")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
