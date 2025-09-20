#!/usr/bin/env python3
"""
Memory optimization script for RAG system
"""

import psutil
import gc
import os
from typing import Dict

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024,
        'total_mb': psutil.virtual_memory().total / 1024 / 1024
    }

def check_memory_threshold(threshold_mb: float = 500) -> bool:
    """Check if memory usage exceeds threshold"""
    memory = get_memory_usage()
    return memory['rss_mb'] > threshold_mb

def force_garbage_collection():
    """Force garbage collection to free memory"""
    gc.collect()
    
def print_memory_status():
    """Print current memory status"""
    memory = get_memory_usage()
    print(f"🧠 Memory Usage:")
    print(f"   RSS: {memory['rss_mb']:.1f} MB")
    print(f"   VMS: {memory['vms_mb']:.1f} MB")
    print(f"   Percent: {memory['percent']:.1f}%")
    print(f"   Available: {memory['available_mb']:.1f} MB")
    print(f"   Total: {memory['total_mb']:.1f} MB")

if __name__ == "__main__":
    print_memory_status()