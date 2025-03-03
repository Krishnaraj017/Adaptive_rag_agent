import streamlit as st
import time
from rag_agent import RagAgent

# Streamlit app title
st.title("RAG Pipeline with Streamlit")
st.subheader("Ask questions about your documents")

# Define default URLs
default_urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://huggingface.co/blog",
    "https://www.promptingguide.ai/"
]

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # # Display the URLs (read-only)
    # st.subheader("Document Sources")
    # for i, url in enumerate(default_urls):
    #     st.text(f"{i+1}. {url}")
    
    initialize_button = st.button("Initialize Agent")

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
    st.session_state.initialized = False

# Initialize the agent when button is clicked
if initialize_button:
    with st.spinner("Initializing RAG Agent..."):
        try:
            agent = RagAgent()
            agent.initialize(default_urls)
            st.session_state.agent = agent
            st.session_state.initialized = True
            st.sidebar.success("Agent initialized successfully!")
        except Exception as e:
            st.sidebar.error(f"Error initializing agent: {str(e)}")

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
