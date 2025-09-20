"""
FastAPI Backend for RAG System
Provides REST API endpoints for document ingestion and querying
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import logging
import os

from src.main_pipeline import RAGPipeline, load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation System API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    filter_metadata: Optional[dict] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[dict]
    confidence_score: float
    source_documents: List[str]
    metadata: dict

class IngestResponse(BaseModel):
    file_path: str
    status: str
    pages_processed: Optional[int] = None
    chunks_created: Optional[int] = None
    embeddings_generated: Optional[int] = None
    doc_ids: Optional[List[str]] = None
    error: Optional[str] = None

class SystemStats(BaseModel):
    vector_database: dict
    config: dict
    status: str

# Initialize RAG pipeline
try:
    config = load_config()
    rag_pipeline = RAGPipeline(config)
    logger.info("RAG pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAG pipeline: {e}")
    rag_pipeline = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG System API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    return {"status": "healthy", "rag_pipeline": "available"}

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a PDF document
    
    Args:
        file: PDF file to upload
        
    Returns:
        IngestResponse with processing results
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
    
    # Check if filename exists and is valid
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    try:
        # Create data directory if it doesn't exist
        data_dir = Path("data/uploads")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file - ensure filename is not None
        filename = file.filename
        if filename is None:
            raise HTTPException(status_code=400, detail="Invalid filename")
            
        file_path = data_dir / filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            
        # Process the document
        result = rag_pipeline.ingest_document(str(file_path))
        
        return IngestResponse(**result)
        
    except Exception as e:
        logger.error(f"Error ingesting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query the document collection
    
    Args:
        request: QueryRequest with question and parameters
        
    Returns:
        QueryResponse with answer and citations
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
    try:
        response = rag_pipeline.query_documents(
            question=request.question,
            top_k=request.top_k or 5
        )
        
        return QueryResponse(
            answer=response.answer,
            citations=response.citations,
            confidence_score=response.confidence_score,
            source_documents=response.source_documents,
            metadata=response.metadata
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all ingested documents"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
    try:
        stats = rag_pipeline.get_system_stats()
        return {
            "total_documents": stats.get('vector_database', {}).get('total_documents', 0),
            "collection_name": stats.get('vector_database', {}).get('collection_name', ''),
            "status": stats.get('status', 'unknown')
        }
    except Exception as e:
        logger.error(f"Error getting document list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
    try:
        stats = rag_pipeline.get_system_stats()
        return SystemStats(**stats)
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the system"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")
        
    try:
        rag_pipeline.vector_store.delete_collection()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
