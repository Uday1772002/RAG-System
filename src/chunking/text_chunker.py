"""
Text Chunking Module
Intelligent text segmentation for RAG systems
"""

import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    page_number: int
    chunk_id: str
    start_char: int
    end_char: int
    metadata: Dict

class TextChunker:
    """Intelligent text chunking with semantic boundaries"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk_document(self, text_pages: List[str], metadata: Dict) -> List[TextChunk]:
        """
        Chunk a document into overlapping segments
        
        Args:
            text_pages: List of text from each page
            metadata: Document metadata
            
        Returns:
            List of TextChunk objects
        """
        chunks = []
        chunk_id = 0
        
        for page_num, page_text in enumerate(text_pages):
            if not page_text.strip():
                continue
                
            # Chunk the page
            page_chunks = self._chunk_page(
                page_text, page_num, chunk_id, metadata
            )
            chunks.extend(page_chunks)
            chunk_id += len(page_chunks)
            
        return chunks
        
    def _chunk_page(self, page_text: str, page_num: int, 
                    start_chunk_id: int, metadata: Dict) -> List[TextChunk]:
        """Chunk a single page of text"""
        chunks = []
        text = page_text.strip()
        
        if len(text) <= self.chunk_size:
            # Page fits in one chunk
            chunk = TextChunk(
                text=text,
                page_number=page_num,
                chunk_id=f"chunk_{start_chunk_id}",
                start_char=0,
                end_char=len(text),
                metadata={
                    **metadata,
                    "page": page_num,
                    "chunk_type": "full_page"
                }
            )
            chunks.append(chunk)
        else:
            # Need to split page into multiple chunks
            page_chunks = self._split_text_with_overlap(text)
            
            for i, (chunk_text, start, end) in enumerate(page_chunks):
                chunk = TextChunk(
                    text=chunk_text,
                    page_number=page_num,
                    chunk_id=f"chunk_{start_chunk_id + i}",
                    start_char=start,
                    end_char=end,
                    metadata={
                        **metadata,
                        "page": page_num,
                        "chunk_type": "partial_page",
                        "chunk_index": i
                    }
                )
                chunks.append(chunk)
                
        return chunks
        
    def _split_text_with_overlap(self, text: str) -> List[tuple]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end >= len(text):
                # Last chunk
                chunk_text = text[start:].strip()
                chunks.append((chunk_text, start, len(text)))
                break
                
            # Find a good break point
            break_point = self._find_break_point(text, start, end)
            
            chunk_text = text[start:break_point].strip()
            chunks.append((chunk_text, start, break_point))
            
            # Move start position with overlap
            start = break_point - self.chunk_overlap
            if start <= 0:
                start = break_point
                
        return chunks
        
    def _find_break_point(self, text: str, start: int, end: int) -> int:
        """Find a good point to break text"""
        # Prefer breaking at sentence boundaries
        sentence_breaks = ['.', '!', '?', '\n\n']
        
        # Look for sentence breaks in the last 200 characters
        search_start = max(start, end - 200)
        search_text = text[search_start:end]
        
        for char in sentence_breaks:
            pos = search_text.rfind(char)
            if pos != -1:
                return search_start + pos + 1
                
        # Look for paragraph breaks
        para_break = search_text.rfind('\n\n')
        if para_break != -1:
            return search_start + para_break + 2
            
        # Look for single line breaks
        line_break = search_text.rfind('\n')
        if line_break != -1:
            return search_start + line_break + 1
            
        # Look for word boundaries
        word_break = search_text.rfind(' ')
        if word_break != -1:
            return search_start + word_break + 1
            
        # If no good break point, just break at the end
        return end
        
    def chunk_by_sentences(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """Alternative chunking method based on sentences"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
