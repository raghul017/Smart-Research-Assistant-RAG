import os
import pickle
import logging
from config import Config

try:
    import sentence_transformers
    import torch
    import transformers
except Exception as e:
    raise ImportError(f"Could not import sentence_transformers or its dependencies: {e}. Please install it with pip install sentence-transformers.")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector storage and retrieval for documents"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
        )
        self.persist_path = os.path.join(Config.CHROMA_PERSIST_DIRECTORY, "faiss_index")
        self.vectorstore = self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """Initialize or load existing vector store"""
        if os.path.exists(self.persist_path):
            with open(self.persist_path, "rb") as f:
                return pickle.load(f)
        else:
            return None

    def _save_vectorstore(self):
        """Persist the vector store"""
        os.makedirs(Config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        with open(self.persist_path, "wb") as f:
            pickle.dump(self.vectorstore, f)
    
    def add_document(self, document_data: Dict[str, Any]) -> bool:
        """
        Add a document to the vector store
        
        Args:
            document_data: Dictionary containing 'content' and 'metadata'
            
        Returns:
            True if successful, False otherwise
        """
        try:
            content = document_data['content']
            metadata = document_data['metadata']
            
            # Split text into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_meta = metadata.copy()
                chunk_meta['chunk_id'] = i
                chunk_meta['chunk_size'] = len(chunk)
                chunk_metadata.append(chunk_meta)
            
            # Add chunks to vector store
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_texts(
                    texts=chunks,
                    embedding=self.embeddings,
                    metadatas=chunk_metadata
                )
            else:
                self.vectorstore.add_texts(
                    texts=chunks,
                    metadatas=chunk_metadata
                )
            
            # Persist the vector store
            self._save_vectorstore()
            
            logger.info(f"Added document {metadata['file_name']} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {str(e)}")
            return False
    
    def search_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents based on query
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            if self.vectorstore is None:
                return []
            if top_k is None:
                top_k = Config.TOP_K_RESULTS
            
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(query, k=top_k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                result = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': float(score)
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents in the vector store"""
        try:
            if self.vectorstore is None:
                return 0
            return len(self.vectorstore.index_to_docstore_id)
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    def delete_document(self, file_path: str) -> bool:
        """
        Delete a document from the vector store
        
        Args:
            file_path: Path of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # FAISS does not support deletion of individual vectors natively.
            # You would need to rebuild the index without the deleted document.
            return False
                
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    def clear_all_documents(self) -> bool:
        """Clear all documents from the vector store"""
        try:
            self.vectorstore = None
            if os.path.exists(self.persist_path):
                os.remove(self.persist_path)
            logger.info("Cleared all documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {str(e)}")
            return False
    
    def get_document_list(self) -> List[Dict[str, Any]]:
        """Get list of all documents in the vector store"""
        try:
            # FAISS does not natively support listing all documents.
            # You can extend this if you store metadata separately.
            return []
            
        except Exception as e:
            logger.error(f"Error getting document list: {str(e)}")
            return [] 