import os
import PyPDF2
from docx import Document
from typing import List, Dict, Any
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles processing of different document types"""
    
    def __init__(self):
        self.supported_extensions = Config.SUPPORTED_EXTENSIONS
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document and extract its content
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document metadata and content
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension not in self.supported_extensions:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Extract text based on file type
            if file_extension == '.pdf':
                content = self._extract_pdf_text(file_path)
            elif file_extension in ['.docx', '.doc']:
                content = self._extract_word_text(file_path)
            elif file_extension == '.txt':
                content = self._extract_text_file(file_path)
            elif file_extension == '.md':
                content = self._extract_text_file(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Create document metadata
            metadata = {
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'file_type': self.supported_extensions[file_extension],
                'file_size': os.path.getsize(file_path),
                'content_length': len(content)
            }
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise
    
    def _extract_word_text(self, file_path: str) -> str:
        """Extract text from Word document"""
        try:
            doc = Document(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting Word text: {str(e)}")
            raise
    
    def _extract_text_file(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error extracting text file: {str(e)}")
            raise
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if file can be processed"""
        if not os.path.exists(file_path):
            return False
        
        file_extension = os.path.splitext(file_path)[1].lower()
        return file_extension in self.supported_extensions
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return list(self.supported_extensions.values()) 