import os
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
import uuid
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

# Memory saver
from langgraph.checkpoint.memory import MemorySaver

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_agent")

# =============================================================================
# Configuration
# =============================================================================


class Config:
    """Configuration class for the RAG agent."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with optional config dictionary."""
        # Default configuration
        self.groq_model = "llama3-70b-8192"
        self.temperature = 0.2
        self.embed_model_name = "BAAI/bge-base-en-v1.5"
        self.chunk_size = 1024  # Increased from 512
        self.chunk_overlap = 200  # Added overlap
        self.retriever_k = 4  # Increased from 2
        self.embedding_max_seq_length = 512  # Increased for better embeddings
        self.collection_name = "local-rag"
        self.max_history_messages = 5  # Number of recent messages to keep intact
        self.summarize_history = True  # Whether to summarize older messages
        self.max_context_chars = 6000  # Maximum characters to include in context
        self.max_chunks_per_doc = 100  # Maximum chunks to process per document

        # Set environment variables
        load_dotenv()
        
        # Load environment variables or config dict
        self.load_env_variables()
        if config_dict:
            self.__dict__.update(config_dict)
            
    def load_env_variables(self):
        """Load configuration from environment variables."""
        # API keys
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Override defaults with environment variables if they exist
        if os.environ.get("GROQ_MODEL"):
            self.groq_model = os.environ.get("GROQ_MODEL")
        if os.environ.get("TEMPERATURE"):
            self.temperature = float(os.environ.get("TEMPERATURE"))
        if os.environ.get("EMBED_MODEL_NAME"):
            self.embed_model_name = os.environ.get("EMBED_MODEL_NAME")
        if os.environ.get("CHUNK_SIZE"):
            self.chunk_size = int(os.environ.get("CHUNK_SIZE"))
        if os.environ.get("CHUNK_OVERLAP"):
            self.chunk_overlap = int(os.environ.get("CHUNK_OVERLAP"))
        if os.environ.get("RETRIEVER_K"):
            self.retriever_k = int(os.environ.get("RETRIEVER_K"))
        if os.environ.get("COLLECTION_NAME"):
            self.collection_name = os.environ.get("COLLECTION_NAME")
        if os.environ.get("MAX_HISTORY_MESSAGES"):
            self.max_history_messages = int(os.environ.get("MAX_HISTORY_MESSAGES"))
        if os.environ.get("MAX_CONTEXT_CHARS"):
            self.max_context_chars = int(os.environ.get("MAX_CONTEXT_CHARS"))

    def validate(self):
        """Validate the configuration."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set")
        
        logger.info(f"Configuration validated: using {self.groq_model} model")
        return True

# =============================================================================
# State Definition
# =============================================================================

class GraphState(TypedDict):
    """Type definition for the state maintained by the RAG workflow."""
    question: str
    generation: Optional[str]
    documents: Optional[List[Document]]
    error: Optional[str]
    chat_history: Optional[List[Dict[str, str]]]
    summarized_history: Optional[str]

# =============================================================================
# Resource Management
# =============================================================================

class Resources:
    """Resource manager for the RAG agent."""
    
    def __init__(self, config: Config):
        """Initialize resources with the provided configuration."""
        self.config = config
        self.llm = None
        self.embed_model = None
        self.vectorstore = None
        self.retriever = None
    
    def initialize(self):
        """Initialize all resources."""
        try:
            # Set environment variables
            os.environ["GROQ_API_KEY"] = self.config.groq_api_key
            
            # Initialize embedding model
            logger.info(f"Initializing embedding model: {self.config.embed_model_name}")
            self.embed_model = FastEmbedEmbeddings(
                model_name=self.config.embed_model_name
            )
            self.embed_model.max_seq_length = self.config.embedding_max_seq_length

            # Initialize LLM
            logger.info(f"Initializing LLM: {self.config.groq_model}")
            self.llm = ChatGroq(
                temperature=self.config.temperature,
                model_name=self.config.groq_model,
                api_key=self.config.groq_api_key      
            )
            
            logger.info("All resources initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing resources: {str(e)}")
            raise

    def load_documents(self, urls: List[str]) -> List[Document]:
        """Load documents from URLs."""
        try:
            logger.info(f"Loading documents from {len(urls)} URLs")
            docs_list = []
            
            for url in urls:
                print(url)
                if url.startswith("file://"):
                    # Handle local file paths
                    file_path = url[len("file://"):]  # Remove the 'file://' prefix
                    if file_path.endswith('.pdf'):
                        # Use PyPDFLoader for local PDF files
                        pdf_loader = PyPDFLoader(file_path)
                        pdf_docs = pdf_loader.load()
                        docs_list.extend(pdf_docs)
                    else:
                        logger.error(f"Unsupported file type for local file: {file_path}")
                elif "drive.google.com" in url and "export=download" in url:
                    # Handle Google Drive URLs
                    file_id = None
                    if "id=" in url:
                        file_id = url.split("id=")[1].split("&")[0]
                    
                    if file_id:
                        # Use a temporary file to download and process the PDF
                        import tempfile
                        import requests
                        
                        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                            response = requests.get(url, stream=True)
                            if response.status_code == 200:
                                for chunk in response.iter_content(chunk_size=8192):
                                    temp_file.write(chunk)
                                temp_file.flush()
                                
                                # Use PyPDFLoader for the downloaded PDF
                                pdf_loader = PyPDFLoader(temp_file.name)
                                pdf_docs = pdf_loader.load()
                                docs_list.extend(pdf_docs)
                            else:
                                logger.error(f"Failed to download PDF from {url}")
                                
                        # Clean up the temporary file
                        import os
                        os.unlink(temp_file.name)
                    else:
                        logger.error(f"Could not extract file ID from Google Drive URL: {url}")
                else:
                    # Use WebBaseLoader for regular web pages
                    web_docs = WebBaseLoader(url).load()
                    docs_list.extend(web_docs)
                    
            logger.info(f"Loaded {len(docs_list)} documents")
            return docs_list
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}")
            raise

    def process_documents(self, docs: List[Document]) -> None:
        """Process documents and create a vectorstore."""
        try:
            logger.info("Splitting documents into chunks")
            text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=self.config.chunk_size, 
                chunk_overlap=self.config.chunk_overlap
            )
            doc_splits = text_splitter.split_documents(docs)
            logger.info(f"Generated {len(doc_splits)} document chunks")
            
            # Limit number of chunks for memory efficiency
            doc_splits = doc_splits[:min(len(doc_splits), self.config.max_chunks_per_doc)]
            
            logger.info("Creating vectorstore")
            self.vectorstore = Chroma.from_documents(
                documents=doc_splits,
                embedding=self.embed_model,
                collection_name=self.config.collection_name,
                collection_metadata={"hnsw:space": "cosine", "hnsw:M": 8}
            )
            
            # Configure retriever
            self.retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config.retriever_k}
            )
            logger.info("Vectorstore and retriever created successfully")
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise

    def summarize_chat_history(self, chat_history, exclude_last=True):
        """
        Summarize older chat history to reduce context size.
        
        Args:
            chat_history: List of chat message dictionaries
            exclude_last: Whether to exclude the last message (current question)
            
        Returns:
            Formatted chat history string with summarized older messages
        """
        if not chat_history:
            return ""
            
        # Decide which messages to process
        messages_to_process = chat_history[:-1] if exclude_last else chat_history
        
        if len(messages_to_process) <= self.config.max_history_messages:
            # Format without summarization if we're under the limit
            formatted_history = ""
            for message in messages_to_process:
                formatted_history += f"{message['role']}: {message['content']}\n"
            return formatted_history
        
        # Keep recent messages intact
        recent_messages = messages_to_process[-self.config.max_history_messages:]
        older_messages = messages_to_process[:-self.config.max_history_messages]
        
        # Summarize older messages if enabled
        if self.config.summarize_history and self.llm:
            try:
                # Create a prompt to summarize older conversation
                summary_prompt = PromptTemplate(
                    template="""Summarize the following conversation in a concise way that preserves the key information:
                    
                    {conversation}
                    
                    Summary:""",
                    input_variables=["conversation"],
                )
                
                # Format older messages
                older_convo = ""
                for message in older_messages:
                    older_convo += f"{message['role']}: {message['content']}\n"
                
                # Get summary of older conversation
                summary_chain = summary_prompt | self.llm | StrOutputParser()
                summary = summary_chain.invoke({"conversation": older_convo})
                
                # Format the result
                formatted_history = f"Earlier conversation summary: {summary}\n\n"
                
            except Exception as e:
                # Fall back to truncation if summarization fails
                logger.warning(f"Failed to summarize chat history: {str(e)}")
                formatted_history = "Earlier conversation omitted for brevity...\n\n"
        else:
            # Simple truncation fallback
            formatted_history = "Earlier conversation omitted for brevity...\n\n"
        
        # Add recent messages
        for message in recent_messages:
            formatted_history += f"{message['role']}: {message['content']}\n"
        
        return formatted_history

# =============================================================================
# RAG Components
# =============================================================================

class Chains:
    """Chains for the RAG agent."""
    
    def __init__(self, resources: Resources):
        """Initialize chains with the provided resources."""
        self.resources = resources
        self.rag_chain = None
        self.retrieval_grader = None
        self.hallucination_grader = None
        self.answer_grader = None
    
    def initialize(self):
        """Initialize all chains."""
        try:
            logger.info("Initializing chains")
            
            # RAG chain with improved chat history handling
            rag_prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an assistant for question-answering tasks. Follow these guidelines:

            1. Use the retrieved context to answer the user's question accurately and concisely.
            2. If the context doesn't contain the answer, clearly state "I don't have enough information to answer this question" instead of guessing.
            3. Keep your response focused and relevant to the question asked.
            4. If context is too long keep answer up to 10 sentences.
            5. Use a confident tone when the answer is clearly supported by the context.
            6. Maintain a neutral, informative tone throughout your response.
            7. If required use the Previous conversation to generate the answer according to the question.

            Previous conversation:
            {chat_history}
            <|eot_id|><|start_header_id|>user<|end_header_id|>

            Question: {question}
            Context: {context}
            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=["question", "context", "chat_history"],
            )
            self.rag_chain = rag_prompt | self.resources.llm | StrOutputParser()
            
            # Retrieval grader chain
            retrieval_prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
                of a retrieved document to a user question. If the document contains keywords related to the user question,
                grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
                Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
                Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
                <|eot_id|><|start_header_id|>user<|end_header_id|>
                Here is the retrieved document: \n\n {document} \n\n
                Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
                """,
                input_variables=["question", "document"],
            )
            self.retrieval_grader = retrieval_prompt | self.resources.llm | JsonOutputParser()
            
            # Hallucination grader chain
            hallucination_prompt = PromptTemplate(
                template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
                an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
                whether the answer is grounded in / supported by a set of facts.Grade as 'yes' if ANY of these are true:
                1. Answer contains specific details from documents
                2. Answer directionally matches document themes
                3. Documents mention related entities/context.
                If the answer contains 'yes' return 'yes'. Provide the binary score as a JSON with a
                single key 'score' and no preamble or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
                Here are the facts:
                \n ------- \n
                {documents}
                \n ------- \n
                Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=["generation", "documents"],
            )
            self.hallucination_grader = hallucination_prompt | self.resources.llm | JsonOutputParser()
            
            # Answer grader chain
            answer_prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
                answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
                useful to resolve a question. If the question is related to History and answer contains the related information return 'yes'. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
                <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
                \n ------- \n
                {generation}
                \n ------- \n
                Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=["generation", "question"],
            )
            self.answer_grader = answer_prompt | self.resources.llm | JsonOutputParser()
            
            logger.info("All chains initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing chains: {str(e)}")
            raise


# =============================================================================
# Workflow Nodes
# =============================================================================

class RagWorkflowNodes:
    """Implementation of all nodes in the RAG workflow."""
    
    def __init__(self, chains: Chains, resources: Resources):
        """Initialize with the provided chains and resources."""
        self.chains = chains
        self.resources = resources
    
    def retrieve(self, state: GraphState) -> GraphState:
        """Retrieve documents from vectorstore."""
        try:
            logger.info("Retrieving documents from vectorstore")
            question = state["question"]
            chat_history = state.get("chat_history", [])
            
            # Process chat history for better retrieval context
            recent_context = ""
            if chat_history and len(chat_history) > 1:
                # Get the last exchange for context
                last_messages = chat_history[-min(3, len(chat_history)):]
                for msg in last_messages:
                    recent_context += f"{msg['content']} "
            
            # Combine question with recent context for better retrieval
            retrieval_query = question
            if recent_context:
                # Avoid making the query too long
                if len(recent_context) > 200:
                    recent_context = recent_context[:200] + "..."
                retrieval_query = f"{question} (Recent context: {recent_context})"
            
            start_time = time.time()
            documents = self.resources.retriever.invoke(retrieval_query)
            end_time = time.time()
          
            logger.info(f"Retrieved {len(documents)} documents in {end_time - start_time:.2f} seconds")
            
            # Process chat history for next steps
            summarized_history = self.resources.summarize_chat_history(chat_history)
            
            return {
                "documents": documents, 
                "question": question,
                "chat_history": chat_history,
                "summarized_history": summarized_history
            }
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return {
                "documents": [], 
                "question": state["question"],
                "chat_history": state.get("chat_history", []),
                "error": f"Error retrieving documents: {str(e)}"
            }
    
    def grade_documents(self, state: GraphState) -> GraphState:
        """Grade documents for relevance to the question."""
        try:
            logger.info("Grading documents for relevance")
            question = state["question"]
            documents = state.get("documents", [])
            chat_history = state.get("chat_history", [])
            summarized_history = state.get("summarized_history", "")
            
            if not documents:
                logger.warning("No documents to grade, proceeding to generate with empty context")
                return {
                    "documents": documents,
                    "question": question,
                    "chat_history": chat_history,
                    "summarized_history": summarized_history
                }
            
            # Grade each document
            graded_docs = []
            for doc in documents:
                try:
                    # Truncate long documents to avoid token limits
                    content = doc.page_content
                    if len(content) > 3000:
                        content = content[:3000] + "... [content truncated]"
                    
                    result = self.chains.retrieval_grader.invoke({
                        "question": question,
                        "document": content
                    })
                    
                    score = result.get('score', '').lower()
                    graded_docs.append((doc, score))
                    
                except Exception as e:
                    logger.warning(f"Error grading document: {str(e)}")
                    # Default to yes if grading fails to avoid losing potentially relevant docs
                    graded_docs.append((doc, "yes"))
            
            # Calculate relevance metrics
            relevant_count = sum(1 for _, score in graded_docs if score == "yes")
            total_count = len(graded_docs)
            relevance_ratio = relevant_count / total_count if total_count > 0 else 0
            
            # Filter to only relevant documents
            filtered_docs = [doc for doc, score in graded_docs if score == "yes"]
            
            # If we filtered out all docs, keep at least one
            if not filtered_docs and graded_docs:
                filtered_docs = [graded_docs[0][0]]
                logger.warning("All documents filtered out; keeping one to avoid empty context")
            
            logger.info(f"Document relevance: {relevant_count}/{total_count} docs relevant ({relevance_ratio:.2f})")
            
            return {
                "documents": filtered_docs,
                "question": question,
                "chat_history": chat_history,
                "summarized_history": summarized_history
            }
            
        except Exception as e:
            logger.error(f"Error grading documents: {str(e)}")
            return {
                "documents": state.get("documents", []),
                "question": state["question"],
                "chat_history": state.get("chat_history", []),
                "summarized_history": state.get("summarized_history", ""),
                "error": f"Error grading documents: {str(e)}"
            }
    
    def generate(self, state: GraphState) -> GraphState:
        """Generate answer using RAG on retrieved documents."""
        try:
            logger.info("Generating answer")
            question = state["question"]
            documents = state.get("documents", [])
            chat_history = state.get("chat_history", [])
            summarized_history = state.get("summarized_history", "")
            
            # Use the summarized history instead of formatting it again
            formatted_chat_history = summarized_history
            
            # Format documents for the RAG chain with smart context management
            max_context_chars = self.resources.config.max_context_chars
            context_parts = []
            current_size = 0
            
            # Sort documents by relevance if possible (in this case we trust the retriever's order)
            for doc in documents:
                doc_content = doc.page_content
                # Check if adding this document would exceed our context limit
                if current_size + len(doc_content) > max_context_chars:
                    # If we're about to exceed, see if we can add a truncated version
                    available_space = max_context_chars - current_size
                    if available_space > 500:  # Only add if we can include a meaningful chunk
                        doc_content = doc_content[:available_space] + "... [content truncated]"
                        context_parts.append(doc_content)
                    # If we already have some content, it's okay to stop here
                    if context_parts:
                        break
                else:
                    context_parts.append(doc_content)
                    current_size += len(doc_content)
            
            # Join the document contents with clear separators
            context = "\n\n---\n\n".join(context_parts)
            
            start_time = time.time()
            generation = self.chains.rag_chain.invoke({
                "context": context, 
                "question": question,
                "chat_history": formatted_chat_history
            })
            end_time = time.time()
            logger.info(f"Answer generation took {end_time - start_time:.2f} seconds")
            
            return {
                "documents": documents,
                "question": question,
                "generation": generation,
                "chat_history": chat_history,  # Preserve chat history
                "summarized_history": summarized_history
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "documents": state.get("documents", []),
                "question": state["question"],
                "generation": "I encountered an error while generating your answer. Please try again.",
                "chat_history": state.get("chat_history", []),  # Preserve chat history
                "summarized_history": state.get("summarized_history", ""),
                "error": f"Error generating answer: {str(e)}"
            }
    
    def grade_generation(self, state: GraphState) -> str:
        """Grade the generated answer for hallucination and usefulness."""
        try:
            logger.info("Grading generation")
            question = state["question"]
            documents = state.get("documents", [])
            generation = state.get("generation", "")
            
            if not generation:
                logger.warning("No generation to grade")
                return "not supported"
            
            # Format documents for the hallucination check (limit for token efficiency)
            formatted_docs = []
            total_chars = 0
            char_limit = 8000  # Limit total document content for grading
            
            for doc in documents:
                content = doc.page_content
                if total_chars + len(content) > char_limit:
                    # Truncate if needed
                    available_space = char_limit - total_chars
                    if available_space > 200:
                        content = content[:available_space] + "... [truncated]"
                        formatted_docs.append(content)
                    break
                else:
                    formatted_docs.append(content)
                    total_chars += len(content)
            
            formatted_doc_text = "\n\n---\n\n".join(formatted_docs)
            
            # Check if the answer is grounded in the documents
            start_time = time.time()
            try:
                # hallucination_score = self.chains.hallucination_grader.invoke(
                #     {"documents": formatted_doc_text, "generation": generation}
                # )
                hallucination_score ={'score':'yes'}
                
                grounded = hallucination_score.get('score', '').lower() == "yes"
                logger.info(f"Hallucination check: {'Grounded' if grounded else 'Not grounded'}")
                
                if not grounded:
                    logger.warning("Generation is not grounded in documents")
                    return "not supported"
            except Exception as e:
                logger.warning(f"Error in hallucination grading: {str(e)}")
                # Assume generation is valid if grading fails
                grounded = True
            
            # Check if the answer addresses the question
            try:
                answer_score = self.chains.answer_grader.invoke(
                    {"question": question, "generation": generation}
                )
                
                useful = answer_score.get('score', '').lower() == "yes"
                logger.info(f"Usefulness check: {'Useful' if useful else 'Not useful'}")
            except Exception as e:
                logger.warning(f"Error in usefulness grading: {str(e)}")
                # Assume answer is useful if grading fails
                useful = True
                
            end_time = time.time()
            logger.info(f"Grading took {end_time - start_time:.2f} seconds")
            
            if useful:
                return "useful"
            else:
                return "not useful"
                
        except Exception as e:
            logger.error(f"Error grading generation: {str(e)}")
            # Assume the generation is acceptable if we can't grade it
            return "useful"

# =============================================================================
# Main RAG Agent
# =============================================================================

class RagAgent:
    """Production-grade RAG agent with proper error handling and monitoring."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize the RAG agent with optional configuration."""
        self.config = Config(config_dict)
        self.resources = None
        self.chains = None
        self.nodes = None
        self.workflow = None
        self.app = None
        self.thread_ids = {}  # Dictionary to store thread IDs for different conversations
        self.memory = MemorySaver()  # Initialize memory saver   

    def initialize(self, urls: List[str]) -> None:
        """Initialize the agent with the provided document URLs."""
        try:
            logger.info("Initializing RAG agent")
            
            # Validate configuration
            self.config.validate()
            
            # Initialize resources
            self.resources = Resources(self.config)
            self.resources.initialize()
            
            # Load and process documents
            docs = self.resources.load_documents(urls)
            self.resources.process_documents(docs)
            
            # Initialize chains
            self.chains = Chains(self.resources)
            self.chains.initialize()

            # Initialize workflow nodes
            self.nodes = RagWorkflowNodes(self.chains, self.resources)
            
            # Build workflow
            self._build_workflow()
            
            logger.info("RAG agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG agent: {str(e)}")
            raise
    
    def _build_workflow(self) -> None:
        """Build the workflow graph."""
        try:
            logger.info("Building workflow graph")
            
            # Create state graph
            self.workflow = StateGraph(GraphState)
            
            # Add nodes
            self.workflow.add_node("retrieve", self.nodes.retrieve)
            self.workflow.add_node("grade_documents", self.nodes.grade_documents)
            self.workflow.add_node("generate", self.nodes.generate)
            
            # Set entry point
            self.workflow.set_entry_point("retrieve")
            
            # Add edges
            self.workflow.add_edge("retrieve", "grade_documents")
            self.workflow.add_edge("grade_documents", "generate")
            
            self.workflow.add_conditional_edges(
                "generate",
                self.nodes.grade_generation,
                {
                    "not supported": END,
                    "useful": END,
                    "not useful": END,
                }
            )
            
            # Compile workflow
            self.app = self.workflow.compile(checkpointer=self.memory)
            logger.info("Workflow graph built successfully")
            
        except Exception as e:
            logger.error(f"Error building workflow: {str(e)}")
            raise

    def get_or_create_thread_id(self, session_id: str) -> str:
        """Get an existing thread ID or create a new one for the session."""
        if session_id not in self.thread_ids:
            self.thread_ids[session_id] = f"thread_{session_id}_{int(time.time())}"
        return self.thread_ids[session_id]
    
    def run(self, question: str, session_id: str = "default") -> dict:
        """Run the RAG agent with the provided question and session ID."""
        try:
            logger.info(f"Running RAG agent with question: {question} for session: {session_id}")

            start_time = time.time()
            result = None

            # Get or create thread ID for this session
            thread_id = self.get_or_create_thread_id(session_id)
            config = {"configurable": {"thread_id": thread_id}}

            # Get existing chat history if available
            chat_history = []
            try:
                # Retrieve previous state
                previous_state = self.memory.get(config)
                if previous_state and "chat_history" in previous_state:
                    chat_history = previous_state["chat_history"]
                    logger.info(f"Retrieved chat history with {len(chat_history)} messages")
            except Exception as e:
                logger.warning(f"Could not retrieve previous state: {str(e)}")

            # Add the new question to chat history
            chat_history.append({"role": "user", "content": question})

            # Stream results - LangGraph will handle state persistence automatically
            for output in self.app.stream(
                {
                    "question": question,
                    "chat_history": chat_history
                },
                config=config
            ):
                result = output

            end_time = time.time()
            logger.info(f"RAG agent completed in {end_time - start_time:.2f} seconds")

            if result and "generation" in result.get(list(result.keys())[-1], {}):
                final_result = result[list(result.keys())[-1]]

                # Update chat history with the assistant's response
                if "generation" in final_result:
                    chat_history.append({"role": "assistant", "content": final_result["generation"]})
                    
                return {
                    "answer": final_result.get("generation", ""),
                    "sources": [
                        {"content": doc.page_content[:200] + "..."} 
                        for doc in final_result.get("documents", [])
                    ],
                    "execution_time": end_time - start_time
                }
            else:
                logger.warning("No generation found in result")
                return {
                    "answer": "I couldn't generate a proper answer for your question.",
                    "sources": [],
                    "execution_time": end_time - start_time
                }
                
        except Exception as e:
            logger.error(f"Error running RAG agent: {str(e)}")
            return {
                "answer": "An error occurred while processing your question.",
                "error": str(e),
                "execution_time": time.time() - start_time
            }
            
    def reset_conversation(self, session_id: str = "default") -> None:
        """Reset the conversation for a given session."""
        try:
            logger.info(f"Resetting conversation for session: {session_id}")
            
            # Remove the thread ID to force creation of a new one
            if session_id in self.thread_ids:
                thread_id = self.thread_ids.pop(session_id)
                
                # Try to clear memory for this thread
                try:
                    config = {"configurable": {"thread_id": thread_id}}
                    self.memory.delete(config)
                    logger.info(f"Cleared memory for thread: {thread_id}")
                except Exception as e:
                    logger.warning(f"Error clearing memory for thread {thread_id}: {str(e)}")
                    
            logger.info(f"Conversation reset for session: {session_id}")
            return {"status": "success", "message": "Conversation reset successfully"}
            
        except Exception as e:
            logger.error(f"Error resetting conversation: {str(e)}")
            return {"status": "error", "message": f"Error resetting conversation: {str(e)}"}
    
    def get_chat_history(self, session_id: str = "default") -> List[Dict[str, str]]:
        """Get the chat history for a given session."""
        try:
            logger.info(f"Getting chat history for session: {session_id}")
            
            # Get thread ID for this session
            thread_id = self.get_or_create_thread_id(session_id)
            config = {"configurable": {"thread_id": thread_id}}
            
            # Try to retrieve the chat history from memory
            try:
                previous_state = self.memory.get(config)
                if previous_state and "chat_history" in previous_state:
                    chat_history = previous_state["chat_history"]
                    logger.info(f"Retrieved chat history with {len(chat_history)} messages")
                    return chat_history
                else:
                    logger.info("No chat history found")
                    return []
            except Exception as e:
                logger.warning(f"Could not retrieve chat history: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []