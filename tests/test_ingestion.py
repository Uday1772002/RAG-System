"""
Tests for the ingestion module
"""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path

from src.ingestion.pdf_processor import PDFProcessor

class TestPDFProcessor:
    """Test cases for PDFProcessor"""
    
    def test_init(self):
        """Test PDFProcessor initialization"""
        processor = PDFProcessor(ocr_enabled=True, ocr_language="eng")
        assert processor.ocr_enabled is True
        assert processor.ocr_language == "eng"
        
    def test_init_defaults(self):
        """Test PDFProcessor initialization with defaults"""
        processor = PDFProcessor()
        assert processor.ocr_enabled is True
        assert processor.ocr_language == "eng"
        
    @patch('src.ingestion.pdf_processor.PyPDF2.PdfReader')
    def test_extract_metadata_success(self, mock_pdf_reader):
        """Test successful metadata extraction"""
        # Mock PDF reader
        mock_reader = Mock()
        mock_reader.metadata = {
            '/Title': 'Test Document',
            '/Author': 'Test Author',
            '/Subject': 'Test Subject',
            '/Creator': 'Test Creator',
            '/Producer': 'Test Producer'
        }
        mock_reader.pages = [Mock()] * 5  # 5 pages
        
        mock_pdf_reader.return_value = mock_reader
        
        processor = PDFProcessor()
        metadata = processor._extract_metadata(Path("test.pdf"))
        
        assert metadata['title'] == 'Test Document'
        assert metadata['author'] == 'Test Author'
        assert metadata['pages'] == 5
        
    def test_needs_ocr_empty_text(self):
        """Test OCR need detection for empty text"""
        processor = PDFProcessor()
        result = processor._needs_ocr([])
        assert result is True
        
    def test_needs_ocr_insufficient_text(self):
        """Test OCR need detection for insufficient text"""
        processor = PDFProcessor()
        text_pages = ["Short", "Very short text", "Minimal"]
        result = processor._needs_ocr(text_pages)
        assert result is True
        
    def test_needs_ocr_sufficient_text(self):
        """Test OCR need detection for sufficient text"""
        processor = PDFProcessor()
        text_pages = ["This is a longer text that should be sufficient for native extraction"]
        result = processor._needs_ocr(text_pages)
        assert result is False

if __name__ == "__main__":
    pytest.main([__file__])
