"""
Test script for LLM integration in RAG system
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.append(str(Path(__file__).parent))

from src.main_pipeline import load_config, RAGPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_llm_integration():
    """Test the LLM integration"""
    try:
        print("🚀 Testing LLM Integration...")
        
        # Load configuration
        config = load_config()
        print(f"✅ Configuration loaded: {config.get('generation', {}).get('provider', 'unknown')} provider")
        
        # Initialize pipeline
        pipeline = RAGPipeline(config)
        print("✅ RAG Pipeline initialized successfully")
        
        # Test with a simple query (without documents for now)
        test_query = "What is artificial intelligence?"
        
        # Create a mock retrieved document for testing
        from src.retrieval.retriever import RetrievalResult
        mock_doc = RetrievalResult(
            text="Artificial intelligence (AI) is a field of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence.",
            page_number=1,
            chunk_id="test_chunk_1",
            similarity_score=0.9,
            metadata={"source": "test_document"},
            source_document="test.pdf"
        )
        
        # Test response generation
        print(f"\n🤖 Testing query: '{test_query}'")
        response = pipeline.response_generator.generate_response(test_query, [mock_doc])
        
        print("\n📝 Generated Response:")
        print("-" * 50)
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Citations: {len(response.citations)}")
        print(f"Provider: {response.metadata.get('provider', 'unknown')}")
        print(f"Model: {response.metadata.get('model', 'unknown')}")
        print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        logger.error(f"Test failed: {e}", exc_info=True)
        return False

def check_environment():
    """Check if environment variables are set up correctly"""
    print("\n🔍 Checking Environment Variables...")
    
    required_vars = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'local': 'LOCAL_MODEL_PATH'
    }
    
    config = load_config()
    provider = config.get('generation', {}).get('provider', 'openai')
    
    print(f"Current provider: {provider}")
    
    if provider in required_vars:
        var_name = required_vars[provider]
        value = os.getenv(var_name)
        if value:
            print(f"✅ {var_name} is set")
            if provider in ['openai', 'anthropic']:
                print(f"   Key preview: {value[:8]}...")
        else:
            print(f"❌ {var_name} is not set")
            print(f"   Please set {var_name} in your .env file")
            return False
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("RAG SYSTEM LLM INTEGRATION TEST")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        print("\n❌ Environment setup incomplete. Please configure your .env file.")
        sys.exit(1)
    
    # Run the test
    success = test_llm_integration()
    
    if success:
        print("\n🎉 LLM Integration test completed successfully!")
        print("\nNext steps:")
        print("1. Add your API key to the .env file")
        print("2. Test with actual PDF documents")
        print("3. Run the full RAG pipeline")
    else:
        print("\n❌ LLM Integration test failed. Check the logs above.")
        sys.exit(1)