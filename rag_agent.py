import os
import time
import logging
import gc
from typing import Dict, List, Optional, Any, Callable, Union
import uuid
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langgraph.graph import END, StateGraph

# For local LLaMA with transformers
import torch
from transformers import pipeline, AutoTokenizer, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

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
        self.llama_model = "meta-llama/Llama-3.2-3B"  # Using a smaller model by default
        self.temperature = 0.2
        self.embed_model_name = "BAAI/bge-base-en-v1.5"
        self.chunk_size = 512
        self.chunk_overlap = 0
        self.retriever_k = 2
        self.embedding_max_seq_length = 128
        self.collection_name = "local-rag"
        self.max_new_tokens = 512  # Control generation length
        self.use_4bit_quantization = True  # Enable 4-bit quantization by default

        # Load environment variables
        load_dotenv()
        self.load_env_variables()
        
        # Override with config dict if provided
        if config_dict:
            self.__dict__.update(config_dict)
            
    def load_env_variables(self):
        """Load configuration from environment variables."""
        if os.environ.get("TEMPERATURE"):
            self.temperature = float(os.environ.get("TEMPERATURE"))
        if os.environ.get("EMBED_MODEL_NAME"):
            self.embed_model_name = os.environ.get("EMBED_MODEL_NAME")
        if os.environ.get("CHUNK_SIZE"):
            self.chunk_size = int(os.environ.get("CHUNK_SIZE"))
        if os.environ.get("RETRIEVER_K"):
            self.retriever_k = int(os.environ.get("RETRIEVER_K"))
        if os.environ.get("COLLECTION_NAME"):
            self.collection_name = os.environ.get("COLLECTION_NAME")
        if os.environ.get("USE_4BIT_QUANTIZATION"):
            self.use_4bit_quantization = os.environ.get("USE_4BIT_QUANTIZATION").lower() == "true"

    def validate(self):
        """Validate the configuration."""
        logger.info(f"Configuration validated: using {self.llama_model} model")
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
        self.pipe = None
    
    def initialize(self):
        """Initialize all resources."""
        try:
            # Initialize embedding model
            logger.info(f"Initializing embedding model: {self.config.embed_model_name}")
            self.embed_model = FastEmbedEmbeddings(
                model_name=self.config.embed_model_name
            )
            self.embed_model.max_seq_length = self.config.embedding_max_seq_length

            # Initialize LLaMA model with Transformers pipeline
            logger.info(f"Initializing local LLaMA model: {self.config.llama_model}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.llama_model)

            # Set a chat template for LLaMA model
            tokenizer.chat_template = """<|im_start|>system
            You are a helpful AI assistant that answers questions based on provided documents.
            <|im_end|>
            <|im_start|>user
            {{query}}
            <|im_end|>
            <|im_start|>assistant
            """
            
            # Configure quantization if enabled
            model_kwargs = {}
            if self.config.use_4bit_quantization:
                logger.info("Using 4-bit quantization for LLaMA model")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                        
            # Create pipeline for text generation with parameters pre-configured
            self.pipe = pipeline(
                "text-generation",
                model=self.config.llama_model,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                tokenizer=tokenizer,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=True if self.config.temperature > 0 else False,
                model_kwargs=model_kwargs
            )
            
            # Create HuggingFacePipeline directly
            self.llm = HuggingFacePipeline(pipeline=self.pipe)
            
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
                        from langchain_community.document_loaders import PyPDFLoader
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
                        from langchain_community.document_loaders import PyPDFLoader
                        
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
                    from langchain_community.document_loaders import WebBaseLoader
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
            doc_splits = doc_splits[:min(len(doc_splits), 100)]
            
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
            
    def cleanup(self):
        """Release resources when done."""
        try:
            logger.info("Cleaning up resources")
            if hasattr(self, 'pipe') and self.pipe is not None:
                del self.pipe
            if hasattr(self, 'llm') and self.llm is not None:
                del self.llm
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            logger.info("Resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")

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
                Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
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
                whether the answer is grounded in / supported by a set of facts. Grade as 'yes' if ANY of these are true:
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
            
            start_time = time.time()
            documents = self.resources.retriever.invoke(question)
            end_time = time.time()
          
            logger.info(f"Retrieved {len(documents)} documents in {end_time - start_time:.2f} seconds")
            
            return {"documents": documents, "question": question}
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return {
                "documents": [], 
                "question": state["question"],
                "error": f"Error retrieving documents: {str(e)}"
            }
    
    def grade_documents(self, state: GraphState) -> GraphState:
        """Grade documents for relevance to the question."""
        try:
            logger.info("Grading documents for relevance")
            question = state["question"]
            documents = state.get("documents", [])
            
            if not documents:
                logger.warning("No documents to grade, proceeding to generate with empty context")
                return {
                    "documents": documents,
                    "question": question
                }
            
            # Grade each document
            graded_docs = []
            for doc in documents:
                try:
                    # Truncate long documents to avoid token limits
                    content = doc.page_content[:10000] if len(doc.page_content) > 10000 else doc.page_content
                    
                    result = self.chains.retrieval_grader.invoke({
                        "question": question,
                        "document": content
                    })
                    
                    score = result.get('score', '').lower()
                    graded_docs.append((doc, score))
                    
                except Exception as e:
                    logger.warning(f"Error grading document: {str(e)}")
                    graded_docs.append((doc, "no"))
            
            # Calculate relevance metrics
            relevant_count = sum(1 for _, score in graded_docs if score == "yes")
            total_count = len(graded_docs)
            relevance_ratio = relevant_count / total_count if total_count > 0 else 0
            
            # Filter to only relevant documents
            filtered_docs = [doc for doc, score in graded_docs if score == "yes"]
            
            # If no relevant documents found, use all documents
            if not filtered_docs and documents:
                logger.warning("No relevant documents found, using all documents")
                filtered_docs = documents
            
            logger.info(f"Document relevance: {relevant_count}/{total_count} docs relevant ({relevance_ratio:.2f})")
            
            return {
                "documents": filtered_docs,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Error grading documents: {str(e)}")
            return {
                "documents": state.get("documents", []),
                "question": state["question"],
                "error": f"Error grading documents: {str(e)}"
            }
    
    def generate(self, state: GraphState) -> GraphState:
        """Generate answer using RAG on retrieved documents."""
        try:
            logger.info("Generating answer")
            question = state["question"]
            documents = state.get("documents", [])
            chat_history = state.get("chat_history", [])
            
            # Format chat history for the prompt - improved implementation
            formatted_chat_history = ""
            if chat_history and len(chat_history) > 1:
                # Skip the current question (last item)
                for message in chat_history[:-1]:
                    role = message["role"]
                    content = message["content"]
                    formatted_chat_history += f"{role}: {content}\n"
            
            # Format documents for the RAG chain
            context = "\n\n".join(doc.page_content for doc in documents)
            
            # Truncate context if too long to fit in context window
            if len(context) > 20000:
                logger.warning("Context too long, truncating to 20000 characters")
                context = context[:20000] + "..."
            
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
                "chat_history": chat_history  # Preserve chat history
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "documents": state.get("documents", []),
                "question": state["question"],
                "generation": "I encountered an error while generating your answer. Please try again.",
                "chat_history": state.get("chat_history", []),
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
            
            # Check if the answer is grounded in the documents
            start_time = time.time()
            
            # Prepare documents content for grading
            docs_content = "\n".join([doc.page_content[:1000] for doc in documents[:3]])
            
            hallucination_score = self.chains.hallucination_grader.invoke(
                {"documents": docs_content, "generation": generation}
            )
            
            grounded = hallucination_score.get('score', '').lower() == "yes"
            logger.info(f"Hallucination check: {'Grounded' if grounded else 'Not grounded'}")
            
            if not grounded:
                logger.warning("Generation is not grounded in documents")
                return "not supported"
            
            # Check if the answer addresses the question
            answer_score = self.chains.answer_grader.invoke(
                {"question": question, "generation": generation}
            )
            end_time = time.time()
            
            useful = answer_score.get('score', '').lower() == "yes"
            logger.info(f"Usefulness check: {'Useful' if useful else 'Not useful'}")
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
                        {
                            "content": doc.page_content[:200] + "...",
                            "metadata": doc.metadata
                        } 
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
    
    def cleanup(self):
        """Clean up resources."""
        if self.resources:
            self.resources.cleanup()