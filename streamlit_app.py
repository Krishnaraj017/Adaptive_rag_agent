import streamlit as st
import time
from rag_agent import RagAgent

# Streamlit app title
st.title("RAG Pipeline with Streamlit")
st.subheader("Ask questions about your documents")
# default_urls = [
#     # NASA Official Sources
#     "https://www.nasa.gov/mission_pages/station/research/index.html",  # ISS Research
#     "https://mars.nasa.gov/mars2020/mission/overview/",  # Mars Exploration Program
    
#     # Scientific Research and Publications
#     "https://www.nature.com/articles/d41586-024-space-section",  # Nature Space Research
#     "https://www.science.org/topic/article-type/research-article/space-astronomy",  # Science Magazine Space Section
    
#     # Space Technology and Innovation
#     "https://www.spacex.com/updates/",  # SpaceX Mission Updates
#     "https://www.blueorigin.com/news/",  # Blue Origin Innovation
    
#     # International Space Agencies
#     "https://www.esa.int/Science_Exploration/Space_Science",  # European Space Agency Research
#     "https://www.isro.gov.in/research-and-development",  # Indian Space Research Organisation
    
#     # Academic and Research Institutions
#     "https://www.jpl.nasa.gov/news",  # Jet Propulsion Laboratory
#     "https://www.seti.org/research",  # SETI Institute Research
    
#     # Planetary and Astronomical Research
#     "https://www.planetary.org/explore/space-topics/",  # The Planetary Society
#     "https://www.space.com/science",  # Space.com Scientific Research
# ]
# Define default URLs
default_urls = [
    # "https://www.gutenberg.org/files/15697/15697-pdf.pdf",
    "https://drive.google.com/uc?export=download&id=1rK8J_I7AyT9sqtHmzl0xrxLtHiuukco9"


    # NASA Official Sources
    # "https://www.nasa.gov/mission_pages/station/research/index.html",  # ISS Research
    # "https://mars.nasa.gov/mars2020/mission/overview/",  # Mars Exploration Program
    
    # # Scientific Research and Publications
    # "https://www.nature.com/articles/d41586-024-space-section",  # Nature Space Research
    # "https://www.science.org/topic/article-type/research-article/space-astronomy",  # Science Magazine Space Section
    
    # # Space Technology and Innovation
    # "https://www.spacex.com/updates/",  # SpaceX Mission Updates
    # "https://www.blueorigin.com/news/",  # Blue Origin Innovation
    
    # # International Space Agencies
    # "https://www.esa.int/Science_Exploration/Space_Science",  # European Space Agency Research
    # "https://www.isro.gov.in/research-and-development",  # Indian Space Research Organisation
    
    # # Academic and Research Institutions
    # "https://www.jpl.nasa.gov/news",  # Jet Propulsion Laboratory
    # "https://www.seti.org/research",  # SETI Institute Research
    
    # # Planetary and Astronomical Research
    # "https://www.planetary.org/explore/space-topics/",  # The Planetary Society
    # "https://www.space.com/science",  # Space.com Scientific Research
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
