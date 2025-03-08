import streamlit as st
import time
from rag_agent import RagAgent
from typing import List
import tempfile
import os

# Set page config for a darker theme
st.set_page_config(page_title="RAG Pipeline", layout="wide")

# Custom CSS to reduce white space and add darker theme
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stApp {
        padding-top: 1rem;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .user-message {
        background-color: #1e3a5f;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 2px;  /* Reduced from 5px */
    }
    .bot-message {
        background-color: #2d3748;
        padding: 8px;
        border-radius: 5px;
        margin-bottom: 2px;  /* Reduced from 5px */
    }
    /* Hide or reduce caption spacing */
    .css-1offfwp {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        font-size: 0.7em !important;
        padding: 0 !important;
    }
    /* Reduce space for horizontal rule */
    hr {
        margin: 5px 0;  /* Reduced from 10px */
        border-color: #4a5568;
    }
    /* Reduce space between expander and surrounding elements */
    .stExpander {
        background-color: #2d3748;
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }
    /* Compact the expander header */
    .streamlit-expanderHeader {
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }
    .stButton button {
        background-color: #4a63a9;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Simplified app title
st.title("RAG Pipeline")

# Default URLs
default_urls = [
    # "https://drive.google.com/uc?export=download&id=1rK8J_I7AyT9sqtHmzl0xrxLtHiuukco9",
]

# Two-column layout
col1, col2 = st.columns([1, 3])

# Configuration in left column
with col1:
    # PDF file uploader
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    # Initialize and clear buttons in a row
    initialize_button = st.button("Initialize Agent")
    # clear_button = st.button("Clear History")
    
    # if clear_button:
    #     st.session_state.conversation_history = []
    #     st.rerun()

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
    st.session_state.initialized = False

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

# Function to handle uploaded PDFs
def handle_uploaded_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[str]:
    temp_file_paths = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_paths.append(f"file://{temp_file.name}")
    return temp_file_paths

# Initialize the agent when button is clicked
if initialize_button:
    with st.spinner("Initializing..."):
        try:
            agent = RagAgent()
            all_urls = default_urls.copy()
            if uploaded_files:
                temp_file_paths = handle_uploaded_files(uploaded_files)
                all_urls.extend(temp_file_paths)
            agent.initialize(all_urls)
            st.session_state.agent = agent
            st.session_state.initialized = True
            st.success("Agent initialized successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            if uploaded_files:
                for temp_file_path in temp_file_paths:
                    if os.path.exists(temp_file_path.replace("file://", "")):
                        os.unlink(temp_file_path.replace("file://", ""))

# Main content in right column
with col2:
    # Display conversation history
    if not st.session_state.conversation_history:
        st.info("Ask a question to begin")
    else:
        conversation_container = st.container()
        with conversation_container:
            for entry in st.session_state.conversation_history:
                # User message - no extra div wrapper
                st.markdown(f"<div class='user-message'><strong>You:</strong> {entry['question']}</div>", unsafe_allow_html=True)
                
                # Agent response - no extra div wrapper
                st.markdown(f"<div class='bot-message'><strong>Bot:</strong> {entry['answer']}</div>", unsafe_allow_html=True)
                
                # Inline execution time with reduced visibility
                st.caption(f"{entry['execution_time']:.2f}s")
                
                # Compact sources display
                # if 'sources' in entry and entry['sources']:
                #     with st.expander("Sources"):
                #         for j, source in enumerate(entry['sources']):
                #             st.markdown(f"**Source {j+1}:** {source['content']}")
                
                st.markdown("<hr>", unsafe_allow_html=True)

    # Simple input form
    with st.form(key="question_form", clear_on_submit=True):
        user_question = st.text_input("Your question:")
        submit_button = st.form_submit_button("Ask")

    # Process question when submitted
    if submit_button and user_question and st.session_state.initialized:
        with st.spinner("Processing..."):
            try:
                start_time = time.time()
                result = st.session_state.agent.run(user_question)
                execution_time = time.time() - start_time
                
                conversation_entry = {
                    'question': user_question,
                    'answer': result['answer'],
                    'execution_time': execution_time,
                    'sources': result.get('sources', [])
                }
                st.session_state.conversation_history.append(conversation_entry)
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif submit_button and user_question and not st.session_state.initialized:
        st.warning("Please initialize the agent first")
