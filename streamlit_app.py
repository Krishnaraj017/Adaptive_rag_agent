import streamlit as st
import time
from rag_agent import RagAgent
from typing import List
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os

# Streamlit app title
st.title("RAG Pipeline with Streamlit")
st.subheader("Ask questions about your documents")

# Default URLs
default_urls = [
    "https://drive.google.com/uc?export=download&id=1rK8J_I7AyT9sqtHmzl0xrxLtHiuukco9",
]

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # PDF file uploader
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    
    initialize_button = st.button("Initialize Agent")

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
    st.session_state.initialized = False

# Function to handle uploaded PDFs and return their temporary file paths
def handle_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    temp_file_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_paths.append(f"file://{temp_file.name}")  # Use file:// to indicate local file paths
    return temp_file_paths

# Initialize the agent when button is clicked
if initialize_button:
    with st.spinner("Initializing RAG Agent..."):
        try:
            agent = RagAgent()
            
            # Combine default URLs with uploaded PDFs (if any)
            all_urls = default_urls.copy()
            if uploaded_files:
                temp_file_paths = handle_uploaded_files(uploaded_files)
                all_urls.extend(temp_file_paths)  # Add temporary file paths to the URLs list
            
            # Initialize the agent with the combined URLs
            agent.initialize(all_urls)
            
            st.session_state.agent = agent
            st.session_state.initialized = True
            st.sidebar.success("Agent initialized successfully!")
        except Exception as e:
            st.sidebar.error(f"Error initializing agent: {str(e)}")
        finally:
            # Clean up temporary files after loading
            if uploaded_files:
                for temp_file_path in temp_file_paths:
                    os.unlink(temp_file_path.replace("file://", ""))

# Main content area
# User input
user_question = st.text_input("Enter your question:")

# Process question when submitted
if user_question and st.session_state.initialized:
    with st.spinner("Processing your question..."):
        try:
            # Run the RAG pipeline with the user's question
            start_time = time.time()
            result = st.session_state.agent.run(user_question)
            execution_time = time.time() - start_time
            
            # Display the answer
            st.header("Answer:")
            st.write(result['answer'])
            
            # Display execution time
            st.caption(f"Execution time: {execution_time:.2f} seconds")
            
            # Display sources
            if 'sources' in result and result['sources']:
                st.subheader("Sources:")
                for i, source in enumerate(result['sources']):
                    with st.expander(f"Source {i+1}"):
                        st.write(source['content'])
                        if 'url' in source:
                            st.markdown(f"[Source Link]({source['url']})")
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
elif user_question and not st.session_state.initialized:
    st.warning("Please initialize the agent first using the sidebar.")

# Footer
st.markdown("---")
st.caption("RAG Pipeline with Streamlit | Developed using Retrieval-Augmented Generation")