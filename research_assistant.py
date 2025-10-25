from document_processor import DocumentProcessor
from vector_store import VectorStore
from llm_manager import LLMManager
from web_search import WebSearch
from typing import List, Dict, Any, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchAssistant:
    """Main research assistant class that orchestrates all components"""
    
    def __init__(self):
        """Initialize the research assistant with all components"""
        try:
            self.document_processor = DocumentProcessor()
            self.vector_store = VectorStore()
            self.llm_manager = LLMManager()
            self.web_search = WebSearch()
            
            logger.info("Research Assistant initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Research Assistant: {str(e)}")
            raise
    
    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """
        Upload and process a document
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with upload status and metadata
        """
        try:
            # Validate file
            if not self.document_processor.validate_file(file_path):
                return {
                    'success': False,
                    'error': 'Unsupported file type or file not found',
                    'file_name': os.path.basename(file_path)
                }
            
            # Process document
            document_data = self.document_processor.process_document(file_path)
            
            # Add to vector store
            success = self.vector_store.add_document(document_data)
            
            if success:
                return {
                    'success': True,
                    'file_name': document_data['metadata']['file_name'],
                    'file_type': document_data['metadata']['file_type'],
                    'file_size': document_data['metadata']['file_size'],
                    'content_length': document_data['metadata']['content_length'],
                    'message': 'Document uploaded and processed successfully'
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to add document to vector store',
                    'file_name': document_data['metadata']['file_name']
                }
                
        except Exception as e:
            logger.error(f"Error uploading document {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_name': os.path.basename(file_path)
            }
    
    def ask_question(self, question: str, use_web_search: bool = False) -> Dict[str, Any]:
        """
        Ask a question and get an answer using RAG
        
        Args:
            question: The question to ask
            use_web_search: Whether to include web search results
            
        Returns:
            Dictionary containing answer and metadata
        """
        try:
            # Search for relevant documents
            context = self.vector_store.search_documents(question)
            
            # Enhance with web search if requested
            if use_web_search and self.web_search.enable_web_search:
                context = self.web_search.enhance_context_with_web(question, context)
            
            # Generate response
            response = self.llm_manager.generate_response(question, context)
            
            return {
                'question': question,
                'answer': response['answer'],
                'sources': response['sources'],
                'context_used': response['context_used'],
                'confidence': response['confidence'],
                'context_chunks': response.get('context_chunks', 0),
                'web_search_used': use_web_search and self.web_search.enable_web_search
            }
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                'question': question,
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again.",
                'sources': [],
                'context_used': False,
                'confidence': 'error',
                'context_chunks': 0,
                'web_search_used': False
            }
    
    def get_document_summary(self, file_path: str) -> Dict[str, Any]:
        """
        Generate a summary of a specific document
        
        Args:
            file_path: Path to the document
            
        Returns:
            Dictionary containing summary and metadata
        """
        try:
            # Process document if not already in vector store
            document_data = self.document_processor.process_document(file_path)
            
            # Generate summary
            summary = self.llm_manager.generate_summary(document_data['content'])
            
            return {
                'success': True,
                'file_name': document_data['metadata']['file_name'],
                'summary': summary,
                'content_length': document_data['metadata']['content_length']
            }
            
        except Exception as e:
            logger.error(f"Error generating summary for {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_name': os.path.basename(file_path)
            }
    
    def generate_study_questions(self, file_path: str, num_questions: int = 5) -> Dict[str, Any]:
        """
        Generate study questions from a document
        
        Args:
            file_path: Path to the document
            num_questions: Number of questions to generate
            
        Returns:
            Dictionary containing questions and metadata
        """
        try:
            # Process document
            document_data = self.document_processor.process_document(file_path)
            
            # Generate questions
            questions = self.llm_manager.generate_questions(document_data['content'], num_questions)
            
            return {
                'success': True,
                'file_name': document_data['metadata']['file_name'],
                'questions': questions,
                'num_questions': len(questions)
            }
            
        except Exception as e:
            logger.error(f"Error generating questions for {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_name': os.path.basename(file_path)
            }
    
    def delete_document(self, file_path: str) -> Dict[str, Any]:
        """
        Delete a document from the system
        
        Args:
            file_path: Path to the document to delete
            
        Returns:
            Dictionary with deletion status
        """
        try:
            success = self.vector_store.delete_document(file_path)
            
            if success:
                return {
                    'success': True,
                    'file_name': os.path.basename(file_path),
                    'message': 'Document deleted successfully'
                }
            else:
                return {
                    'success': False,
                    'error': 'Document not found or could not be deleted',
                    'file_name': os.path.basename(file_path)
                }
                
        except Exception as e:
            logger.error(f"Error deleting document {file_path}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'file_name': os.path.basename(file_path)
            }
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """
        Get list of all uploaded documents
        
        Returns:
            List of document metadata
        """
        try:
            return self.vector_store.get_document_list()
        except Exception as e:
            logger.error(f"Error getting document list: {str(e)}")
            return []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        
        Returns:
            Dictionary containing system statistics
        """
        try:
            document_count = self.vector_store.get_document_count()
            document_list = self.vector_store.get_document_list()
            
            total_size = sum(doc.get('file_size', 0) for doc in document_list)
            
            return {
                'total_documents': len(document_list),
                'total_chunks': document_count,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'supported_formats': self.document_processor.get_supported_formats(),
                'web_search_enabled': self.web_search.enable_web_search
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {
                'error': str(e)
            }
    
    def clear_all_documents(self) -> Dict[str, Any]:
        """
        Clear all documents from the system
        
        Returns:
            Dictionary with operation status
        """
        try:
            success = self.vector_store.clear_all_documents()
            
            if success:
                return {
                    'success': True,
                    'message': 'All documents cleared successfully'
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to clear documents'
                }
                
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            } 