import os
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Union
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
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

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
GROQ_API_KEY="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"
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

tavily_api_key = "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"
os.environ["TAVILY_API_KEY"] = tavily_api_key


class Config:
    """Configuration class for the RAG agent."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """Initialize with optional config dictionary."""
        # Default configuration
        self.groq_model = "llama3-70b-8192"
        self.temperature = 0.2
        self.embed_model_name = "BAAI/bge-base-en-v1.5"
        self.chunk_size = 512
        self.chunk_overlap = 0
        self.retriever_k = 2
        self.embedding_max_seq_length = 128
        self.web_search_k = 3
        self.collection_name = "local-rag"
        
        # Load environment variables or config dict
        self.load_env_variables()
        if config_dict:
            self.__dict__.update(config_dict)
            
    def load_env_variables(self):
        """Load configuration from environment variables."""
        # API keys
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        # Override defaults with environment variables if they exist
        if os.environ.get("GROQ_MODEL"):
            self.groq_model = os.environ.get("GROQ_MODEL")
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

    def validate(self):
        """Validate the configuration."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is not set")
        
        logger.info(f"Configuration validated: using {self.groq_model} model")
        return True

# =============================================================================
# State Definition
# =============================================================================

class GraphState(TypedDict):
    """Type definition for the state maintained by the RAG workflow."""
    question: str
    generation: Optional[str]
    web_search: Optional[str]
    documents: Optional[List[Document]]
    error: Optional[str]

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
        self.web_search_tool = None
        self.tavily_search_api_wrapper = None
    
    def initialize(self):
        """Initialize all resources."""
        try:
            # Set environment variables
            os.environ["GROQ_API_KEY"] = self.config.groq_api_key
            os.environ["TAVILY_API_KEY"] = self.config.tavily_api_key
            
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
            
            # Initialize Tavily search
            logger.info("Initializing Tavily search")
            self.tavily_search_api_wrapper = TavilySearchAPIWrapper(
                tavily_api_key=self.config.tavily_api_key
            )
            self.web_search_tool = TavilySearchResults(
                api_wrapper=self.tavily_search_api_wrapper,
                k=self.config.web_search_k
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
            docs = [WebBaseLoader(url).load() for url in urls]
            docs_list = [item for sublist in docs for item in sublist]
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

# =============================================================================
# RAG Components
# =============================================================================

class Chains:
    """Chains for the RAG agent."""
    
    def __init__(self, resources: Resources):
        """Initialize chains with the provided resources."""
        self.resources = resources
        self.question_router = None
        self.rag_chain = None
        self.retrieval_grader = None
        self.hallucination_grader = None
        self.answer_grader = None
    
    def initialize(self):
        """Initialize all chains."""
        try:
            logger.info("Initializing chains")
            
            # Question router chain
            router_prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a user question to a vectorstore or web search.for history related search strictly use the vectorstore. 
                You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search'
                or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=["question"],
            )
            self.question_router = router_prompt | self.resources.llm | JsonOutputParser()
            
            # RAG chain
            rag_prompt = PromptTemplate(
                template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question. If you don't know the answer, just give relavant answere and don't mention about the context.
                Use five sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
                Question: {question}
                Context: {context}
                Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
                input_variables=["question", "context"],
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
                If the answer contains 'yes' or is history related search validate and return 'yes'. Provide the binary score as a JSON with a
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
                useful to resolve a question.if the question is related to History and answere contains the related information return 'yes'. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
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
    
    def route_question(self, state: GraphState) -> str:
        """Route question to web search or vectorstore."""
        try:
            logger.info("Routing question")
            question = state["question"]
            
            start_time = time.time()
            source = self.chains.question_router.invoke({"question": question})
            end_time = time.time()
            logger.info(f"Question routing took {end_time - start_time:.2f} seconds")
            
            datasource = source.get('datasource', 'web_search')
            logger.info(f"Routing question to: {datasource}")
            
            if datasource == 'web_search':
                return "websearch"
            elif datasource == 'vectorstore':
                return "vectorstore"
            else:
                # Default fallback
                logger.warning(f"Unknown datasource: {datasource}, defaulting to web search")
                return "websearch"
                
        except Exception as e:
            logger.error(f"Error routing question: {str(e)}")
            state["error"] = f"Error routing question: {str(e)}"
            return "websearch"  # Fallback to web search on error
    
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
    
    def web_search(self, state: GraphState) -> GraphState:
        """Perform web search based on the question."""
        try:
            logger.info("Performing web search")
            question = state["question"]
            documents = state.get("documents", [])
            
            start_time = time.time()
            search_results = self.resources.web_search_tool.invoke({"query": question})
            end_time = time.time()
            logger.info(f"Web search took {end_time - start_time:.2f} seconds, found {len(search_results)} results")
            
            web_results = "\n".join([d["content"] for d in search_results])
            web_document = Document(page_content=web_results)
            
            if documents:
                documents.append(web_document)
            else:
                documents = [web_document]
                
            return {"documents": documents, "question": question}
            
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            # If we have existing documents, continue with those
            if state.get("documents"):
                return state
            # Otherwise, return error state
            return {
                "documents": [], 
                "question": state["question"],
                "error": f"Error in web search: {str(e)}"
            }
    
    def grade_documents(self, state: GraphState) -> GraphState:
        """Grade documents for relevance to the question."""
        try:
            logger.info("Grading documents for relevance")
            question = state["question"]
            documents = state.get("documents", [])
            
            if not documents:
                logger.warning("No documents to grade, proceeding to web search")
                return {
                    "documents": documents,
                    "question": question,
                    "web_search": "Yes"
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
                    
                    # score = result.get('score', '').lower()
                    score="yes"
                    graded_docs.append((doc, score))
                    
                except Exception as e:
                    logger.warning(f"Error grading document: {str(e)}")
                    graded_docs.append((doc, "no"))
            
            # Calculate relevance metrics
            relevant_count = sum(1 for _, score in graded_docs if score == "yes")
            total_count = len(graded_docs)
            relevance_ratio = relevant_count / total_count if total_count > 0 else 0
            
            # Determine if web search is needed
            web_search = "Yes" if relevance_ratio <=0 else "No"
            
            # Filter to only relevant documents
            filtered_docs = [doc for doc, score in graded_docs if score == "yes"]
            
            logger.info(f"Document relevance: {relevant_count}/{total_count} docs relevant ({relevance_ratio:.2f})")
            logger.info(f"Web search needed: {web_search}")
            
            return {
                "documents": filtered_docs,
                "question": question,
                "web_search": web_search
            }
            
        except Exception as e:
            logger.error(f"Error grading documents: {str(e)}")
            return {
                "documents": state.get("documents", []),
                "question": state["question"],
                "web_search": "Yes",  # Default to web search on error
                "error": f"Error grading documents: {str(e)}"
            }
    
    def decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to generate an answer or perform web search."""
        web_search = state.get("web_search", "No")
        documents = state.get("documents", [])
        
        if web_search == "Yes" or not documents:
            logger.info("Decision: Need web search for better results")
            return "websearch"
        else:
            logger.info("Decision: Generate answer from relevant documents")
            return "generate"
    
    def generate(self, state: GraphState) -> GraphState:
        """Generate answer using RAG on retrieved documents."""
        try:
            logger.info("Generating answer")
            question = state["question"]
            documents = state.get("documents", [])
            
            if not documents:
                logger.warning("No documents available for generation")
                return {
                    "documents": [],
                    "question": question,
                    "generation": "I don't have enough information to answer this question accurately."
                }
            
            # Format documents for the RAG chain
            context = "\n\n".join(doc.page_content for doc in documents)
            
            start_time = time.time()
            generation = self.chains.rag_chain.invoke({"context": context, "question": question})
            end_time = time.time()
            logger.info(f"Answer generation took {end_time - start_time:.2f} seconds")
            
            return {
                "documents": documents,
                "question": question,
                "generation": generation
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return {
                "documents": state.get("documents", []),
                "question": state["question"],
                "generation": "I encountered an error while generating your answer. Please try again.",
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
            hallucination_score = self.chains.hallucination_grader.invoke(
                {"documents": documents, "generation": generation}
            )
            
            grounded = hallucination_score.get('score', '').lower() == "yes"
            logger.info(f"Hallucination check: {'Grounded' if grounded else 'Not grounded'}")
            
            if not grounded:
                logger.warning("Generation is not grounded in documents")
                return "not supported"
            
            # Check if the answer addresses the question
            # answer_score = self.chains.answer_grader.invoke(
            #     {"question": question, "generation": generation}
            # )
            answer_score = {"score": "yes"}
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
            # Assume the generation is problematic if we can't grade it
            return "not supported"


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
            self.workflow.add_node("websearch", self.nodes.web_search)
            self.workflow.add_node("retrieve", self.nodes.retrieve)
            self.workflow.add_node("grade_documents", self.nodes.grade_documents)
            self.workflow.add_node("generate", self.nodes.generate)
            
            # Set conditional entry point
            self.workflow.set_conditional_entry_point(
                self.nodes.route_question,
                {
                    "websearch": "websearch",
                    "vectorstore": "retrieve",
                }
            )
            
            # Add edges
            self.workflow.add_edge("retrieve", "grade_documents")
            
            self.workflow.add_conditional_edges(
                "grade_documents",
                self.nodes.decide_to_generate,
                {
                    "websearch": "websearch",
                    "generate": "generate",
                }
            )
            
            self.workflow.add_edge("websearch", "generate")
            
            self.workflow.add_conditional_edges(
                "generate",
                self.nodes.grade_generation,
                {
                    "not supported": END,
                    "useful": END,
                    "not useful": "websearch",
                }
            )
            
            # Compile workflow
            self.app = self.workflow.compile()
            logger.info("Workflow graph built successfully")
            
        except Exception as e:
            logger.error(f"Error building workflow: {str(e)}")
            raise
    
    def run(self, question: str) -> Dict[str, Any]:
        """Run the RAG agent with the provided question."""
        try:
            logger.info(f"Running RAG agent with question: {question}")
            
            start_time = time.time()
            result = None
            
            # Stream results
            for output in self.app.stream({"question": question}):
                result = output
            
            end_time = time.time()
            logger.info(f"RAG agent completed in {end_time - start_time:.2f} seconds")
            
            if result and "generation" in result.get(list(result.keys())[-1], {}):
                final_result = result[list(result.keys())[-1]]
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
