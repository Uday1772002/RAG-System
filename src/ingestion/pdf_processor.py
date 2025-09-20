"""
PDF Processing Module
Handles both native and scanned PDF documents
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import PyPDF2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Processes PDF documents with support for native and scanned content"""
    
    def __init__(self, ocr_enabled: bool = True, ocr_language: str = "eng"):
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language
        
    def process_pdf(self, file_path: str) -> Dict:
        """
        Process a PDF file and extract text content
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted text, metadata, and page info
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"PDF file not found: {file_path}")
                
            # Extract basic metadata
            metadata = self._extract_metadata(file_path)
            
            # Try native text extraction first
            native_text = self._extract_native_text(file_path)
            
            # If native text is insufficient, use OCR
            if self.ocr_enabled and self._needs_ocr(native_text):
                ocr_text = self._extract_ocr_text(file_path)
                # Combine native and OCR text
                final_text = self._merge_texts(native_text, ocr_text)
            else:
                final_text = native_text
                
            return {
                "file_path": str(file_path),
                "metadata": metadata,
                "text": final_text,
                "pages": len(final_text),
                "processing_method": "native" if not self._needs_ocr(native_text) else "hybrid"
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
            
    def _extract_metadata(self, file_path: Path) -> Dict:
        """Extract PDF metadata"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info = pdf_reader.metadata
                
                return {
                    "title": info.get('/Title', ''),
                    "author": info.get('/Author', ''),
                    "subject": info.get('/Subject', ''),
                    "creator": info.get('/Creator', ''),
                    "producer": info.get('/Producer', ''),
                    "pages": len(pdf_reader.pages),
                    "file_size": file_path.stat().st_size
                }
        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
            return {}
            
    def _extract_native_text(self, file_path: Path) -> List[str]:
        """Extract text from native PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_pages = []
                
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    text_pages.append(text.strip())
                    
                return text_pages
        except Exception as e:
            logger.warning(f"Could not extract native text: {e}")
            return []
            
    def _needs_ocr(self, text_pages: List[str]) -> bool:
        """Determine if OCR is needed based on text quality"""
        if not text_pages:
            return True
            
        # Check if text is mostly empty or contains mostly whitespace
        total_chars = sum(len(page) for page in text_pages)
        if total_chars < 100:  # Very little text
            return True
            
        # Check for common OCR indicators
        for page in text_pages:
            if len(page.strip()) < 50:  # Very short pages
                continue
            # If page has reasonable text length, assume native extraction worked
            if len(page.strip()) > 100:
                return False
                
        return True
        
    def _extract_ocr_text(self, file_path: Path) -> List[str]:
        """Extract text using OCR"""
        try:
            # Convert PDF to images
            images = convert_from_path(file_path, dpi=300)
            ocr_text = []
            
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1} with OCR")
                
                # Preprocess image for better OCR
                processed_image = self._preprocess_image(image)
                
                # Extract text using Tesseract
                text = pytesseract.image_to_string(
                    processed_image, 
                    lang=self.ocr_language,
                    config='--psm 6'  # Assume uniform block of text
                )
                
                ocr_text.append(text.strip())
                
            return ocr_text
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return []
            
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
            
        # Enhance contrast
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        return image
        
    def _merge_texts(self, native_text: List[str], ocr_text: List[str]) -> List[str]:
        """Merge native and OCR text, preferring better quality"""
        if not ocr_text:
            return native_text
        if not native_text:
            return ocr_text
            
        merged = []
        max_pages = max(len(native_text), len(ocr_text))
        
        for i in range(max_pages):
            native = native_text[i] if i < len(native_text) else ""
            ocr = ocr_text[i] if i < len(ocr_text) else ""
            
            # Choose the better text based on length and quality
            if len(native.strip()) > len(ocr.strip()) * 1.5:
                merged.append(native)
            else:
                merged.append(ocr)
                
        return merged
