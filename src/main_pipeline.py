"""
Main RAG Pipeline
Orchestrates the entire RAG system workflow
"""

import logging
import os
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Fix relative imports
try:
    from .ingestion.pdf_processor import PDFProcessor
    from .chunking.text_chunker import TextChunker
    from .embeddings.embedding_generator import EmbeddingGenerator, VectorStore
    from .retrieval.retriever import DocumentRetriever
    from .generation.response_generator import ResponseGenerator, GeneratedResponse
except ImportError:
    from src.ingestion.pdf_processor import PDFProcessor
    from src.chunking.text_chunker import TextChunker
    from src.embeddings.embedding_generator import EmbeddingGenerator, VectorStore
    from src.retrieval.retriever import DocumentRetriever
    from src.generation.response_generator import ResponseGenerator, GeneratedResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Main pipeline for the RAG system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pdf_processor = PDFProcessor(
            ocr_enabled=config.get('ocr', {}).get('enabled', True),
            ocr_language=config.get('ocr', {}).get('language', 'eng')
        )
        self.chunker = TextChunker(
            chunk_size=config.get('ingestion', {}).get('chunk_size', 1000),
            chunk_overlap=config.get('ingestion', {}).get('chunk_overlap', 200)
        )
        self.embedding_generator = EmbeddingGenerator(
            model_name=config.get('embeddings', {}).get('model', 'all-MiniLM-L6-v2'),
            device=config.get('embeddings', {}).get('device', 'cpu')
        )
        self.vector_store = VectorStore(
            persist_directory=config.get('vector_db', {}).get('persist_directory', './data/vector_db')
        )
        self.retriever = DocumentRetriever(self.vector_store, self.embedding_generator)
        
        # Initialize response generator with config
        generation_config = config.get('generation', {})
        self.response_generator = ResponseGenerator(generation_config)
        
    def ingest_document(self, file_path: str) -> Dict:
        """
        Ingest a PDF document
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Starting ingestion of: {file_path}")
            
            # 1. Process PDF
            pdf_result = self.pdf_processor.process_pdf(file_path)
            logger.info(f"PDF processed: {pdf_result['pages']} pages")
            
            # 2. Chunk text
            chunks = self.chunker.chunk_document(
                pdf_result['text'], 
                pdf_result['metadata']
            )
            logger.info(f"Text chunked into {len(chunks)} chunks")
            
            # 3. Generate embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedding_generator.generate_embeddings(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # 4. Store in vector database
            chunk_dicts = []
            for chunk in chunks:
                chunk_dict = {
                    'chunk_id': chunk.chunk_id,
                    'page_number': chunk.page_number,
                    'text': chunk.text,
                    'metadata': {
                        **chunk.metadata,
                        'file_path': file_path
                    }
                }
                chunk_dicts.append(chunk_dict)
                
            doc_ids = self.vector_store.add_documents(chunk_dicts, embeddings)
            logger.info(f"Stored {len(doc_ids)} chunks in vector database")
            
            return {
                'file_path': file_path,
                'status': 'success',
                'pages_processed': pdf_result['pages'],
                'chunks_created': len(chunks),
                'embeddings_generated': len(embeddings),
                'doc_ids': doc_ids
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return {
                'file_path': file_path,
                'status': 'error',
                'error': str(e)
            }
            
    def query_documents(self, question: str, top_k: int = 5) -> GeneratedResponse:
        """
        Query the document collection
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Returns:
            GeneratedResponse with answer and citations
        """
        try:
            logger.info(f"Processing query: {question}")
            
            # 1. Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # 2. Generate response
            response = self.response_generator.generate_response(question, retrieved_docs)
            logger.info("Response generated successfully")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
            
    def get_system_stats(self) -> Dict:
        """Get system statistics"""
        try:
            vector_stats = self.vector_store.get_collection_stats()
            
            return {
                'vector_database': vector_stats,
                'config': self.config,
                'status': 'operational'
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'status': 'error', 'error': str(e)}
            
    def batch_ingest(self, file_paths: List[str]) -> List[Dict]:
        """
        Ingest multiple documents in batch
        
        Args:
            file_paths: List of PDF file paths
            
        Returns:
            List of ingestion results
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.ingest_document(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in batch ingestion for {file_path}: {e}")
                results.append({
                    'file_path': file_path,
                    'status': 'error',
                    'error': str(e)
                })
                
        return results

def load_config() -> Dict:
    """Load configuration from file"""
    try:
        import yaml
        
        config_path = Path("config/settings.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default configuration
            return {
                'ingestion': {'chunk_size': 1000, 'chunk_overlap': 200},
                'ocr': {'enabled': True, 'language': 'eng'},
                'embeddings': {'model': 'all-MiniLM-L6-v2', 'device': 'cpu'},
                'vector_db': {'persist_directory': './data/vector_db'},
                'generation': {'model': 'gpt-3.5-turbo'}
            }
    except ImportError:
        # Return default configuration if yaml not available
        return {
            'ingestion': {'chunk_size': 1000, 'chunk_overlap': 200},
            'ocr': {'enabled': True, 'language': 'eng'},
            'embeddings': {'model': 'all-MiniLM-L6-v2', 'device': 'cpu'},
            'vector_db': {'persist_directory': './data/vector_db'},
            'generation': {'model': 'gpt-3.5-turbo'}
        }

if __name__ == "__main__":
    # Example usage
    config = load_config()
    pipeline = RAGPipeline(config)
    
    # Example: Ingest a document
    # result = pipeline.ingest_document("path/to/document.pdf")
    # print(result)
    
    # Example: Query documents
    # response = pipeline.query_documents("What is the main topic?")
    # print(response.answer)
