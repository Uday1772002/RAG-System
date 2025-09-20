"""
Response Generation Module with FREE LLM Options
Supports Ollama (local), Hugging Face (free), and mock implementations
"""

import logging
import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

# Free LLM imports
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Fix relative imports
try:
    from ..retrieval.retriever import RetrievalResult
except ImportError:
    from src.retrieval.retriever import RetrievalResult

logger = logging.getLogger(__name__)

@dataclass
class GeneratedResponse:
    """Represents a generated response with citations"""
    answer: str
    citations: List[Dict]
    confidence_score: float
    source_documents: List[str]
    metadata: Dict

class FreeResponseGenerator:
    """Generates responses using FREE LLM options"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.provider = self.config.get('provider', 'ollama')
        self.model = self.config.get('model', 'llama2')
        self.max_tokens = self.config.get('max_tokens', 1000)
        self.temperature = self.config.get('temperature', 0.7)
        
        # Initialize the selected LLM
        self.llm = None
        self._initialize_llm()
        
    def _initialize_llm(self):
        """Initialize the selected FREE LLM provider"""
        try:
            if self.provider == "ollama":
                if not OLLAMA_AVAILABLE:
                    logger.error("Ollama not available. Install with: pip install ollama")
                    logger.info("Or install Ollama from: https://ollama.ai/")
                    self.llm = "mock"
                    return
                    
                # Test Ollama connection
                try:
                    ollama.list()  # Test if Ollama is running
                    self.llm = "ollama"
                    logger.info(f"✅ Ollama initialized with model: {self.model}")
                except Exception as e:
                    logger.error(f"Ollama not running. Start with: ollama serve")
                    self.llm = "mock"
                    
            elif self.provider == "huggingface":
                if not TRANSFORMERS_AVAILABLE:
                    logger.error("Transformers not available. Install with: pip install transformers torch")
                    self.llm = "mock"
                    return
                
                # Use a lightweight free model
                model_name = self.config.get('hf_model', 'microsoft/DialoGPT-small')
                try:
                    self.hf_pipeline = pipeline(
                        "text-generation",
                        model=model_name,
                        device="cpu",  # Use CPU for compatibility
                        torch_dtype=torch.float32
                    )
                    self.llm = "huggingface"
                    logger.info(f"✅ Hugging Face model initialized: {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load HF model: {e}")
                    self.llm = "mock"
                    
            else:
                logger.warning(f"Unknown provider '{self.provider}'. Using mock.")
                self.llm = "mock"
                
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = "mock"
            
    def generate_response(self, query: str, retrieved_docs: List[RetrievalResult]) -> GeneratedResponse:
        """Generate response using the configured LLM"""
        try:
            if not retrieved_docs:
                return self._generate_no_results_response(query)
                
            # Prepare context
            context = self._prepare_context(retrieved_docs)
            
            # Generate response using the appropriate LLM
            if self.llm == "ollama":
                answer = self._generate_with_ollama(query, context)
            elif self.llm == "huggingface":
                answer = self._generate_with_huggingface(query, context)
            else:
                answer = self._generate_mock_response(query, context)
            
            # Extract citations and calculate confidence
            citations = self._extract_citations(answer, retrieved_docs)
            confidence = self._calculate_confidence(retrieved_docs, answer)
            source_docs = list(set(doc.source_document for doc in retrieved_docs))
            
            return GeneratedResponse(
                answer=answer,
                citations=citations,
                confidence_score=confidence,
                source_documents=source_docs,
                metadata={
                    "provider": self.provider,
                    "model": self.model,
                    "num_sources": len(retrieved_docs),
                    "generation_method": "free_llm"
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_error_response(query, str(e))
    
    def _generate_with_ollama(self, query: str, context: str) -> str:
        """Generate response using Ollama (free local LLM)"""
        try:
            prompt = f"""Based on the following context, answer the question concisely and accurately.

Context:
{context}

Question: {query}

Answer:"""

            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return self._generate_mock_response(query, context)
    
    def _generate_with_huggingface(self, query: str, context: str) -> str:
        """Generate response using Hugging Face transformers (free)"""
        try:
            # Keep prompt short for small models
            prompt = f"Context: {context[:500]}...\nQ: {query}\nA:"
            
            outputs = self.hf_pipeline(
                prompt,
                max_length=min(len(prompt.split()) + 100, 512),  # Conservative limit
                num_return_sequences=1,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.hf_pipeline.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            # Extract only the answer part
            answer = generated_text[len(prompt):].strip()
            
            return answer if answer else "I found relevant information but couldn't generate a complete response."
            
        except Exception as e:
            logger.error(f"Hugging Face generation failed: {e}")
            return self._generate_mock_response(query, context)
    
    def _generate_mock_response(self, query: str, context: str) -> str:
        """Generate a mock response when LLM is not available"""
        return f"""Based on the provided documents, I found information related to your question: "{query}"

The retrieved passages contain relevant content that may help answer your query. However, I'm currently running in mock mode because no LLM provider is properly configured.

To get actual AI-generated responses, please:
1. Install Ollama (free): https://ollama.ai/
2. Or configure Hugging Face models
3. Or add API keys for paid services

For now, please review the source documents I've retrieved for you."""

    def _prepare_context(self, retrieved_docs: List[RetrievalResult]) -> str:
        """Prepare context string for LLM"""
        context_parts = []
        
        for i, doc in enumerate(retrieved_docs[:3]):  # Limit to top 3 for token efficiency
            context_part = f"Source {i+1} (Page {doc.page_number}):\n{doc.text[:500]}...\n"
            context_parts.append(context_part)
            
        return "\n".join(context_parts)
    
    def _extract_citations(self, answer: str, retrieved_docs: List[RetrievalResult]) -> List[Dict]:
        """Extract citations from retrieved documents"""
        citations = []
        
        for i, doc in enumerate(retrieved_docs):
            citation = {
                "source": doc.source_document,
                "page": doc.page_number,
                "chunk_id": doc.chunk_id,
                "text_snippet": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text,
                "similarity_score": doc.similarity_score,
                "citation_id": f"cite_{i+1}"
            }
            citations.append(citation)
            
        return citations
    
    def _calculate_confidence(self, retrieved_docs: List[RetrievalResult], answer: str) -> float:
        """Calculate confidence score"""
        if not retrieved_docs:
            return 0.0
            
        avg_similarity = sum(doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs)
        source_boost = min(len(retrieved_docs) * 0.1, 0.3)
        length_boost = min(len(answer) / 1000, 0.2)
        
        confidence = avg_similarity + source_boost + length_boost
        return min(confidence, 1.0)
    
    def _generate_no_results_response(self, query: str) -> GeneratedResponse:
        """Response when no documents found"""
        answer = f"I couldn't find any relevant information in the available documents to answer your question: '{query}'. Please try rephrasing your question or check if the relevant documents have been uploaded."
        
        return GeneratedResponse(
            answer=answer,
            citations=[],
            confidence_score=0.0,
            source_documents=[],
            metadata={
                "provider": self.provider,
                "model": self.model,
                "num_sources": 0,
                "generation_method": "no_results"
            }
        )
    
    def _generate_error_response(self, query: str, error: str) -> GeneratedResponse:
        """Response when generation fails"""
        answer = f"I encountered an error while processing your question: '{query}'. Error: {error}"
        
        return GeneratedResponse(
            answer=answer,
            citations=[],
            confidence_score=0.0,
            source_documents=[],
            metadata={
                "provider": self.provider,
                "model": self.model,
                "num_sources": 0,
                "generation_method": "error",
                "error": error
            }
        )

# Backward compatibility
ResponseGenerator = FreeResponseGenerator