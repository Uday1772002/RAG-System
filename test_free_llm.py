#!/usr/bin/env python3
"""
Test script for FREE LLM integration
Tests different LLM providers without requiring API keys
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.generation.response_generator import ResponseGenerator, GeneratedResponse
from src.retrieval.retriever import RetrievalResult

def test_mock_llm():
    """Test mock LLM (always works)"""
    print("🧪 Testing Mock LLM...")
    
    config = {
        'provider': 'mock',
        'model': 'mock-model',
        'max_tokens': 500,
        'temperature': 0.7
    }
    
    generator = ResponseGenerator(config)
    
    # Create mock retrieved documents
    mock_docs = [
        RetrievalResult(
            text="This is a sample document about artificial intelligence and machine learning.",
            page_number=1,
            chunk_id="chunk_1",
            similarity_score=0.9,
            metadata={"topic": "AI"},
            source_document="sample.pdf"
        ),
        RetrievalResult(
            text="Machine learning is a subset of AI that focuses on algorithms.",
            page_number=2,
            chunk_id="chunk_2",
            similarity_score=0.8,
            metadata={"topic": "ML"},
            source_document="sample.pdf"
        )
    ]
    
    # Test response generation
    response = generator.generate_response("What is machine learning?", mock_docs)
    
    print(f"✅ Mock Response Generated:")
    print(f"   Provider: {response.metadata.get('provider', 'unknown')}")
    print(f"   LLM Status: {response.metadata.get('llm_status', 'unknown')}")
    print(f"   Confidence: {response.confidence_score:.2f}")
    print(f"   Sources: {len(response.source_documents)}")
    print(f"   Answer: {response.answer[:200]}...")
    print()
    
    return True

def test_ollama_llm():
    """Test Ollama LLM (requires Ollama to be running)"""
    print("🦙 Testing Ollama LLM...")
    
    config = {
        'provider': 'ollama',
        'model': 'llama2',
        'max_tokens': 500,
        'temperature': 0.7
    }
    
    try:
        generator = ResponseGenerator(config)
        
        # Create test query and documents
        mock_docs = [
            RetrievalResult(
                text="Python is a high-level programming language known for its simplicity and readability.",
                page_number=1,
                chunk_id="chunk_1",
                similarity_score=0.95,
                metadata={"topic": "Python"},
                source_document="python_guide.pdf"
            )
        ]
        
        response = generator.generate_response("What is Python?", mock_docs)
        
        print(f"✅ Ollama Response Generated:")
        print(f"   Provider: {response.metadata.get('provider', 'unknown')}")
        print(f"   LLM Status: {response.metadata.get('llm_status', 'unknown')}")
        print(f"   Model: {response.metadata.get('model', 'unknown')}")
        print(f"   Confidence: {response.confidence_score:.2f}")
        print(f"   Answer: {response.answer[:300]}...")
        print()
        
        return response.metadata.get('llm_status') == 'ollama'
        
    except Exception as e:
        print(f"❌ Ollama test failed: {e}")
        return False

def test_huggingface_llm():
    """Test Hugging Face LLM (free but requires model download)"""
    print("🤗 Testing Hugging Face LLM...")
    
    config = {
        'provider': 'huggingface',
        'hf_model': 'microsoft/DialoGPT-small',  # Small model for testing
        'max_tokens': 100,
        'temperature': 0.7
    }
    
    try:
        generator = ResponseGenerator(config)
        
        # Create simple test
        mock_docs = [
            RetrievalResult(
                text="The weather today is sunny and warm.",
                page_number=1,
                chunk_id="chunk_1",
                similarity_score=0.9,
                metadata={"topic": "Weather"},
                source_document="weather.pdf"
            )
        ]
        
        response = generator.generate_response("How is the weather?", mock_docs)
        
        print(f"✅ Hugging Face Response:")
        print(f"   Provider: {response.metadata.get('provider', 'unknown')}")
        print(f"   LLM Status: {response.metadata.get('llm_status', 'unknown')}")
        print(f"   Confidence: {response.confidence_score:.2f}")
        print(f"   Answer: {response.answer[:200]}...")
        print()
        
        return response.metadata.get('llm_status') == 'huggingface'
        
    except Exception as e:
        print(f"❌ Hugging Face test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing FREE LLM Integration for RAG System")
    print("=" * 50)
    
    results = {}
    
    # Test 1: Mock LLM (always works)
    results['mock'] = test_mock_llm()
    
    # Test 2: Ollama (requires installation and running)
    results['ollama'] = test_ollama_llm()
    
    # Test 3: Hugging Face (requires model download)
    results['huggingface'] = test_huggingface_llm()
    
    # Summary
    print("📊 Test Results Summary:")
    print("-" * 30)
    for provider, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {provider.title()}: {status}")
    
    print("\n🎯 Recommendations:")
    if results['ollama']:
        print("   ✅ Ollama is working! This is the best free option.")
    elif results['huggingface']:
        print("   ✅ Hugging Face is working! Good free alternative.")
    else:
        print("   📖 To enable free LLM:")
        print("      1. Install Ollama: https://ollama.ai/")
        print("      2. Run: ollama serve")
        print("      3. Download model: ollama pull llama2")
    
    print("\n✨ Your RAG system is ready to use with FREE LLMs!")

if __name__ == "__main__":
    main()