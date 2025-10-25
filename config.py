import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for the Research Assistant"""
    
    # API Keys
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
    
    # Vector Database
    CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # Model Configuration
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-pro")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
    
    # Document Processing
    SUPPORTED_EXTENSIONS = {
        '.pdf': 'PDF',
        '.txt': 'Text',
        '.docx': 'Word',
        '.doc': 'Word',
        '.md': 'Markdown'
    }
    
    # Chunking Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Search Configuration
    TOP_K_RESULTS = 5
    SIMILARITY_THRESHOLD = 0.7
    
    # Web Search Configuration
    ENABLE_WEB_SEARCH = bool(os.getenv("ENABLE_WEB_SEARCH", "False"))
    MAX_WEB_RESULTS = 3 