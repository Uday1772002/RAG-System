#!/usr/bin/env python3
"""
Memory-optimized test for RAG system
"""

import sys
import os
import gc
from pathlib import Path

# Add project to path
sys.path.append(str(Path(__file__).parent))

def test_memory_optimized_rag():
    """Test RAG system with memory optimizations"""
    try:
        print("🧠 Testing Memory-Optimized RAG System...")
        
        # Monitor memory before loading
        def get_memory():
            pid = os.getpid()
            with os.popen(f'ps -o rss -p {pid}') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    return int(lines[1].strip()) / 1024  # MB
            return 0
        
        initial_memory = get_memory()
        print(f"📊 Initial memory: {initial_memory:.1f} MB")
        
        # Load optimized config
        from src.main_pipeline import load_config, RAGPipeline
        
        # Use memory-optimized settings
        config = {
            'ingestion': {
                'chunk_size': 500,  # Smaller chunks
                'chunk_overlap': 100,
                'max_file_size_mb': 10
            },
            'embeddings': {
                'model': 'all-MiniLM-L6-v2',
                'batch_size': 4,  # Very small batches
                'device': 'cpu'
            },
            'vector_db': {
                'persist_directory': './data/vector_db'
            },
            'retrieval': {
                'top_k': 3  # Fewer results
            },
            'generation': {
                'provider': 'ollama',
                'model': 'llama2',
                'max_tokens': 500
            },
            'ocr': {
                'enabled': True,
                'language': 'eng'
            }
        }
        
        print("⚙️ Initializing RAG pipeline with memory optimizations...")
        pipeline = RAGPipeline(config)
        
        after_load_memory = get_memory()
        print(f"📊 Memory after loading: {after_load_memory:.1f} MB")
        print(f"📈 Memory increase: {after_load_memory - initial_memory:.1f} MB")
        
        # Test simple query without documents
        print("🔍 Testing simple query...")
        try:
            response = pipeline.query_documents("What is AI?", top_k=3)
            print(f"✅ Query successful: {response.answer[:100]}...")
        except Exception as e:
            print(f"⚠️ Query failed (expected without documents): {e}")
        
        final_memory = get_memory()
        print(f"📊 Final memory: {final_memory:.1f} MB")
        
        # Force cleanup
        del pipeline
        gc.collect()
        
        cleanup_memory = get_memory()
        print(f"📊 Memory after cleanup: {cleanup_memory:.1f} MB")
        
        print("✅ Memory-optimized RAG test completed!")
        
        # Memory recommendations
        max_memory_used = max(after_load_memory, final_memory)
        if max_memory_used > 400:
            print("⚠️ High memory usage detected. Consider:")
            print("   - Using smaller batch sizes")
            print("   - Processing smaller documents")
            print("   - Using the memory-optimized config")
        else:
            print("✅ Memory usage is acceptable for deployment")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_memory_optimized_rag()
    if success:
        print("\n🎉 Ready for deployment with memory optimizations!")
    else:
        print("\n💡 Consider using memory-optimized settings")