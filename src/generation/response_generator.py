import logging
from typing import List, Dict
from dataclasses import dataclass

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from ..retrieval.retriever import RetrievalResult
except ImportError:
    from src.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class GeneratedResponse:
    answer: str
    citations: List[Dict]
    confidence_score: float
    source_documents: List[str]
    metadata: Dict

class ResponseGenerator:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.provider = self.config.get('provider', 'mock')
        self.model = self.config.get('model', 'llama2')
        self.llm_status = "mock"
        
        if self.provider == "ollama" and OLLAMA_AVAILABLE:
            try:
                ollama.list()
                self.llm_status = "ollama"
                logger.info(f"Ollama ready with {self.model}")
            except:
                logger.error("Ollama not running")
        
    def generate_response(self, query, retrieved_docs):
        if not retrieved_docs:
            return GeneratedResponse(
                answer=f"No documents found for: {query}",
                citations=[],
                confidence_score=0.0,
                source_documents=[],
                metadata={"provider": self.provider}
            )
        
        answer = f"Based on the documents, here's what I found for '{query}': The retrieved content contains relevant information."
        
        citations = []
        for i, doc in enumerate(retrieved_docs):
            citations.append({
                "source": doc.source_document,
                "page": doc.page_number,
                "text_snippet": doc.text[:200] + "...",
                "similarity_score": doc.similarity_score
            })
        
        return GeneratedResponse(
            answer=answer,
            citations=citations,
            confidence_score=0.8,
            source_documents=[doc.source_document for doc in retrieved_docs],
            metadata={"provider": self.provider}
        )

    def format_response_with_citations(self, response):
        formatted = response.answer
        if response.citations:
            formatted += "\n\nSources:\n"
            for citation in response.citations:
                formatted += f"- {citation['source']} (Page {citation['page']})\n"
        return formatted
