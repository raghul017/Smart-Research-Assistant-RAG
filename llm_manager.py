import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import PromptTemplate
from typing import List, Dict, Any, Optional
import logging
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    """Manages interactions with the Gemini LLM"""
    
    def __init__(self):
        if not Config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required. Please set it in your .env file.")
        
        # Configure Gemini
        genai.configure(api_key=Config.GOOGLE_API_KEY)
        
        # Initialize the LLM
        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_output_tokens=Config.MAX_TOKENS,
            google_api_key=Config.GOOGLE_API_KEY,
            convert_system_message_to_human=True
        )
        
        # Define system prompt for research assistant
        self.system_prompt = """You are a helpful research assistant for students. Your role is to:

1. Answer questions based on the provided context from the student's documents
2. Provide accurate, well-structured responses
3. Cite specific sources when possible
4. Be educational and explain concepts clearly
5. If the context doesn't contain enough information, acknowledge this and suggest what additional information might be needed
6. Use a friendly, encouraging tone suitable for students

When responding:
- Always base your answers on the provided context first
- If you need to make assumptions, clearly state them
- Provide examples when helpful
- Encourage further questions and exploration

Remember: You're here to help students learn and understand their course materials better."""

        # RAG prompt template
        self.rag_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
Context from your documents:
{context}

Question: {question}

Please answer the question based on the context provided above. If the context doesn't contain enough information to fully answer the question, acknowledge this and suggest what additional information might be helpful.

Answer:"""
        )
    
    def generate_response(self, question: str, context: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate a response using RAG approach
        
        Args:
            question: User's question
            context: List of relevant document chunks
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            if not context:
                # No context provided, generate general response
                messages = [
                    SystemMessage(content=self.system_prompt),
                    HumanMessage(content=question)
                ]
                
                response = self.llm.invoke(messages)
                
                return {
                    'answer': response.content,
                    'sources': [],
                    'context_used': False,
                    'confidence': 'low'
                }
            
            # Prepare context for RAG
            context_text = self._prepare_context(context)
            
            # Create prompt with context
            prompt = self.rag_prompt_template.format(
                context=context_text,
                question=question
            )
            
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Extract sources from context
            sources = self._extract_sources(context)
            
            return {
                'answer': response.content,
                'sources': sources,
                'context_used': True,
                'confidence': self._calculate_confidence(context),
                'context_chunks': len(context)
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try again.",
                'sources': [],
                'context_used': False,
                'confidence': 'error'
            }
    
    def _prepare_context(self, context: List[Dict[str, Any]]) -> str:
        """Prepare context text from document chunks"""
        context_parts = []
        
        for i, chunk in enumerate(context, 1):
            source = chunk['metadata'].get('file_name', 'Unknown source')
            content = chunk['content']
            similarity = chunk.get('similarity_score', 0)
            
            context_parts.append(f"Source {i} ({source}, relevance: {similarity:.3f}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _extract_sources(self, context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context"""
        sources = []
        
        for chunk in context:
            metadata = chunk['metadata']
            source = {
                'file_name': metadata.get('file_name', 'Unknown'),
                'file_type': metadata.get('file_type', 'Unknown'),
                'similarity_score': chunk.get('similarity_score', 0),
                'chunk_id': metadata.get('chunk_id', 0)
            }
            sources.append(source)
        
        return sources
    
    def _calculate_confidence(self, context: List[Dict[str, Any]]) -> str:
        """Calculate confidence level based on context quality"""
        if not context:
            return 'low'
        
        # Calculate average similarity score
        avg_similarity = sum(chunk.get('similarity_score', 0) for chunk in context) / len(context)
        
        if avg_similarity > 0.8:
            return 'high'
        elif avg_similarity > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def generate_summary(self, content: str) -> str:
        """Generate a summary of the provided content"""
        try:
            summary_prompt = f"""
Please provide a concise summary of the following content, highlighting the key points and main ideas:

{content}

Summary:"""
            
            messages = [
                SystemMessage(content="You are a helpful assistant that creates clear, concise summaries."),
                HumanMessage(content=summary_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Unable to generate summary due to an error."
    
    def generate_questions(self, content: str, num_questions: int = 5) -> List[str]:
        """Generate study questions based on the content"""
        try:
            questions_prompt = f"""
Based on the following content, generate {num_questions} thoughtful study questions that would help a student understand and engage with the material:

{content}

Please provide the questions in a numbered list format:"""
            
            messages = [
                SystemMessage(content="You are a helpful educational assistant that creates engaging study questions."),
                HumanMessage(content=questions_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the response to extract questions
            lines = response.content.strip().split('\n')
            questions = []
            
            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('Q')):
                    # Remove numbering and clean up
                    question = line.split('.', 1)[-1].strip()
                    if question.startswith('Q'):
                        question = question.split(' ', 1)[-1].strip()
                    questions.append(question)
            
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return ["Unable to generate questions due to an error."] 