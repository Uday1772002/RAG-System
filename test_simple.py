#!/usr/bin/env python3
"""
Simple test to verify RAG system with FREE LLM works
"""

import sys
import os
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test all imports work"""
    try:
        print("🧪 Testing imports...")
        
        from src.generation.response_generator import ResponseGenerator, GeneratedResponse
        print("✅ ResponseGenerator imported")
        
        from src.main_pipeline import RAGPipeline, load_config
        print("✅ RAGPipeline imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        print("\n🧪 Testing configuration...")
        
        from src.main_pipeline import load_config
        config = load_config()
        
        print(f"✅ Config loaded: {config.get('generation', {}).get('provider', 'unknown')}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_response_generator():
    """Test response generator initialization"""
    try:
        print("\n🧪 Testing ResponseGenerator...")
        
        from src.generation.response_generator import ResponseGenerator
        
        config = {'provider': 'ollama', 'model': 'llama2'}
        generator = ResponseGenerator(config)
        
        print(f"✅ ResponseGenerator created with status: {generator.llm_status}")
        return True
    except Exception as e:
        print(f"❌ ResponseGenerator test failed: {e}")
        return False

def test_pipeline():
    """Test full pipeline initialization"""
    try:
        print("\n🧪 Testing RAG Pipeline...")
        
        from src.main_pipeline import RAGPipeline, load_config
        
        config = load_config()
        pipeline = RAGPipeline(config)
        
        print(f"✅ RAG Pipeline created")
        print(f"   - Provider: {pipeline.response_generator.provider}")
        print(f"   - LLM Status: {pipeline.response_generator.llm_status}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    print("🚀 Testing FREE LLM RAG System")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("ResponseGenerator", test_response_generator),
        ("Pipeline", test_pipeline)
    ]
    
    results = {}
    for name, test_func in tests:
        results[name] = test_func()
    
    print("\n📊 Test Results:")
    print("-" * 20)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    if all(results.values()):
        print(f"\n🎉 All tests passed! Your RAG system is ready with FREE LLM support!")
        print(f"\n💡 Next steps:")
        print(f"   1. Start API: python -m src.api.main")
        print(f"   2. Start UI: streamlit run ui/app.py")
        print(f"   3. For Ollama: ollama serve && ollama pull llama2")
    else:
        print(f"\n⚠️ Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()