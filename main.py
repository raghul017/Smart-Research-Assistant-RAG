import streamlit as st
import os
import tempfile
from research_assistant import ResearchAssistant
import pandas as pd
from datetime import datetime
import time

#Page configuration
st.set_page_config(
    page_title="Personal Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and dark backgrounds
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        background-color: #232946 !important;
        color: #eaeaea !important;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #1f77b4 !important;
        color: #fff !important;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #393552 !important;
        color: #eaeaea !important;
        border-left: 4px solid #9c27b0;
    }
    .stats-card {
        background-color: #232946 !important;
        color: #eaeaea !important;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        background-color: #232946 !important;
        color: #eaeaea !important;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
    .stExpander, .stExpander > div {
        background-color: #232946 !important;
        color: #eaeaea !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'research_assistant' not in st.session_state:
    try:
        st.session_state.research_assistant = ResearchAssistant()
        st.session_state.chat_history = []
        st.session_state.uploaded_files = []
    except Exception as e:
        st.error(f"Failed to initialize Research Assistant: {str(e)}")
        st.stop()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

def main():
    # Header
    st.markdown('<h1 class="main-header">📚 Personal Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your AI-powered study companion using RAG with Gemini</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings & Tools")
        
        # System Stats
        st.subheader("📊 System Statistics")
        stats = st.session_state.research_assistant.get_system_stats()
        
        if 'error' not in stats:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", stats['total_documents'])
                st.metric("Total Size", f"{stats['total_size_mb']} MB")
            with col2:
                st.metric("Chunks", stats['total_chunks'])
                st.metric("Web Search", "✅" if stats['web_search_enabled'] else "❌")
        
        # Document Management
        st.subheader("📁 Document Management")
        
        # Upload files
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'txt', 'docx', 'doc', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX, DOC, MD"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in [f['name'] for f in st.session_state.uploaded_files]:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Process the file
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        result = st.session_state.research_assistant.upload_document(tmp_path)
                        
                        if result['success']:
                            st.session_state.uploaded_files.append({
                                'name': uploaded_file.name,
                                'path': tmp_path,
                                'size': uploaded_file.size,
                                'type': uploaded_file.type
                            })
                            st.success(f"✅ {uploaded_file.name} uploaded successfully!")
                        else:
                            st.error(f"❌ Failed to upload {uploaded_file.name}: {result.get('error', 'Unknown error')}")
                    
                    # Clean up temporary file
                    os.unlink(tmp_path)
        
        # Document list
        st.subheader("📋 Your Documents")
        documents = st.session_state.research_assistant.get_document_list()
        
        if documents:
            for doc in documents:
                with st.expander(f"📄 {doc['file_name']}"):
                    st.write(f"**Type:** {doc['file_type']}")
                    st.write(f"**Size:** {doc['file_size']} bytes")
                    st.write(f"**Chunks:** {doc['chunk_count']}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"🗑️ Delete", key=f"delete_{doc['file_name']}"):
                            # Note: This would need the actual file path to delete
                            st.warning("Delete functionality requires file path mapping")
                    with col2:
                        if st.button(f"📝 Summary", key=f"summary_{doc['file_name']}"):
                            st.info("Summary feature would be implemented here")
        else:
            st.info("No documents uploaded yet. Upload some files to get started!")
        
        # Clear all documents
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.research_assistant.clear_all_documents()['success']:
                st.session_state.uploaded_files = []
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Question input
        question = st.text_area(
            "Ask a question about your uploaded documents:",
            placeholder="e.g., What are the main concepts discussed in my notes?",
            height=100
        )
        
        # Options
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            use_web_search = st.checkbox("🔍 Include web search results", value=False)
        with col1_2:
            if st.button("🚀 Ask Question", type="primary", use_container_width=True):
                if question.strip():
                    ask_question(question, use_web_search)
                else:
                    st.warning("Please enter a question.")
        
        # Chat history
        st.subheader("💭 Conversation History")
        
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {chat['question']}
                </div>
                """, unsafe_allow_html=True)
                
                # Assistant message
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {chat['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if chat.get('sources'):
                    with st.expander(f"📚 Sources ({len(chat['sources'])})"):
                        for j, source in enumerate(chat['sources']):
                            st.write(f"**{j+1}. {source['file_name']}**")
                            st.write(f"   Type: {source['file_type']}")
                            st.write(f"   Relevance: {source['similarity_score']:.3f}")
                
                # Show metadata
                with st.expander("ℹ️ Response Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Confidence:** {chat['confidence']}")
                    with col2:
                        st.write(f"**Context Used:** {'✅' if chat['context_used'] else '❌'}")
                    with col3:
                        st.write(f"**Web Search:** {'✅' if chat.get('web_search_used') else '❌'}")
                
                st.divider()
        else:
            st.info("No conversation history yet. Ask a question to get started!")
    
    with col2:
        st.header("🛠️ Quick Tools")
        
        # Document summary tool
        st.subheader("📝 Document Summary")
        if documents:
            selected_doc = st.selectbox(
                "Select a document to summarize:",
                options=[doc['file_name'] for doc in documents],
                key="summary_select"
            )
            
            if st.button("📋 Generate Summary", key="generate_summary"):
                # This would need the actual file path
                st.info("Summary generation would be implemented here")
        else:
            st.info("Upload documents to use this feature")
        
        # Study questions tool
        st.subheader("❓ Study Questions")
        if documents:
            selected_doc_questions = st.selectbox(
                "Select a document for questions:",
                options=[doc['file_name'] for doc in documents],
                key="questions_select"
            )
            
            num_questions = st.slider("Number of questions:", 3, 10, 5)
            
            if st.button("🎯 Generate Questions", key="generate_questions"):
                # This would need the actual file path
                st.info("Question generation would be implemented here")
        else:
            st.info("Upload documents to use this feature")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Refresh System", key="refresh"):
            st.rerun()
        
        if st.button("📊 Export Chat", key="export"):
            export_chat_history()
        
        # Help section
        st.subheader("❓ Help")
        st.markdown("""
        1. **Upload Documents**: Use the sidebar to upload your study materials (PDF, DOCX, TXT, etc.)
        2. **Ask Questions**: Type questions about your documents in the main area
        3. **Get Answers**: The AI will search through your documents and provide contextual answers
        4. **Use Tools**: Generate summaries and study questions for your documents
        5. **Web Search**: Enable web search for additional information (optional)
        """)

def ask_question(question: str, use_web_search: bool):
    """Process a question and add to chat history"""
    with st.spinner("🤔 Thinking..."):
        response = st.session_state.research_assistant.ask_question(question, use_web_search)
        
        # Add to chat history
        st.session_state.chat_history.append({
            'timestamp': datetime.now(),
            'question': question,
            'answer': response['answer'],
            'sources': response.get('sources', []),
            'context_used': response.get('context_used', False),
            'confidence': response.get('confidence', 'unknown'),
            'web_search_used': response.get('web_search_used', False)
        })
        
        st.rerun()

def export_chat_history():
    """Export chat history to CSV"""
    if st.session_state.chat_history:
        df = pd.DataFrame(st.session_state.chat_history)
        csv = df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Chat History",
            data=csv,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.warning("No chat history to export")

if __name__ == "__main__":
    main() 