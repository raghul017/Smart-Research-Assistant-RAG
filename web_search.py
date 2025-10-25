import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearch:
    """Handles web search functionality for additional context"""
    
    def __init__(self):
        self.serpapi_key = Config.SERPAPI_API_KEY
        self.enable_web_search = Config.ENABLE_WEB_SEARCH
        self.max_results = Config.MAX_WEB_RESULTS
    
    def search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the web for additional information
        
        Args:
            query: Search query
            
        Returns:
            List of web search results
        """
        if not self.enable_web_search:
            return []
        
        try:
            if self.serpapi_key:
                return self._search_with_serpapi(query)
            else:
                return self._search_with_google(query)
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
    
    def _search_with_serpapi(self, query: str) -> List[Dict[str, Any]]:
        """Search using SerpAPI"""
        try:
            url = "https://serpapi.com/search"
            params = {
                'q': query,
                'api_key': self.serpapi_key,
                'num': self.max_results
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if 'organic_results' in data:
                for result in data['organic_results'][:self.max_results]:
                    web_result = {
                        'title': result.get('title', ''),
                        'snippet': result.get('snippet', ''),
                        'link': result.get('link', ''),
                        'source': 'web_search'
                    }
                    results.append(web_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error with SerpAPI search: {str(e)}")
            return []
    
    def _search_with_google(self, query: str) -> List[Dict[str, Any]]:
        """Basic web scraping search (fallback)"""
        try:
            # This is a simplified approach - in production, you'd want to use a proper search API
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results (this is a basic implementation)
            search_results = soup.find_all('div', class_='g')
            
            for result in search_results[:self.max_results]:
                title_elem = result.find('h3')
                snippet_elem = result.find('span', class_='st')
                link_elem = result.find('a')
                
                if title_elem and snippet_elem:
                    web_result = {
                        'title': title_elem.get_text().strip(),
                        'snippet': snippet_elem.get_text().strip(),
                        'link': link_elem.get('href', '') if link_elem else '',
                        'source': 'web_search'
                    }
                    results.append(web_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error with Google search: {str(e)}")
            return []
    
    def extract_content_from_url(self, url: str) -> Optional[str]:
        """
        Extract text content from a URL
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted text content or None if failed
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text[:2000]  # Limit content length
            
        except Exception as e:
            logger.error(f"Error extracting content from URL {url}: {str(e)}")
            return None
    
    def enhance_context_with_web(self, query: str, existing_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance existing context with web search results
        
        Args:
            query: Original query
            existing_context: Existing document context
            
        Returns:
            Enhanced context with web results
        """
        if not self.enable_web_search:
            return existing_context
        
        web_results = self.search_web(query)
        enhanced_context = existing_context.copy()
        
        for result in web_results:
            # Extract content from the URL if possible
            if result.get('link'):
                content = self.extract_content_from_url(result['link'])
                if content:
                    result['content'] = content
                else:
                    result['content'] = result.get('snippet', '')
            else:
                result['content'] = result.get('snippet', '')
            
            # Add to enhanced context
            enhanced_context.append({
                'content': result['content'],
                'metadata': {
                    'file_name': result['title'],
                    'file_type': 'Web Result',
                    'source': 'web_search',
                    'url': result.get('link', '')
                },
                'similarity_score': 0.5  # Default score for web results
            })
        
        return enhanced_context 