# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
# from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper # Correct import
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_groq import ChatGroq
# import os
# userdata = {"GROQ_API_KEY": "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"}
# GROQ_API_KEY = userdata.get("GROQ_API_KEY")
# os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# tavily_api_key = "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"
# os.environ["TAVILY_API_KEY"] = tavily_api_key

# # Instantiate the embed model
# embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# # Instantiate the ChatGroq LLM.
# llm = ChatGroq(model="llama3-70b-8192")

# # Instantiate the TavilySearchAPIWrapper.
# tavily_search_api_wrapper = TavilySearchAPIWrapper(tavily_api_key=tavily_api_key)

# # Create the TavilySearchResults tool.
# # tools = [tool]

# # Instantiate and override the default LLM with updated values.
# llm = ChatGroq(temperature=0, model_name="Llama3-8b-8192", api_key=userdata.get("GROQ_API_KEY"))

# urls = [
#     "https://lilianweng.github.io/posts/2023-06-23-agent/",
#     # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
#     # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
#     # "https://gautam75.medium.com/ten-ways-to-serve-large-language-models-a-comprehensive-guide-292250b02c11",
#     # "https://ai.googleblog.com/",
#     # "https://huggingface.co/blog",
#     # # "https://openai.com/research",
#     # # "https://www.deepmind.com/blog",
#     # "https://arxiv.org/list/cs.AI/recent",
#     # "https://www.promptingguide.ai/",
#     # "https://blog.eleuther.ai/",
#     # # "https://cleverhans.ai/",
#     # "https://robust-ml.github.io/",
#     # "https://www.anyscale.com/blog",
#     # "https://python.langchain.com/docs/",
#     # "https://www.partnershiponai.org/resources/",
#     # "https://ainowinstitute.org/publications.html"

# ]

# docs = [WebBaseLoader(url).load() for url in urls]
# print(docs)
# docs_list = [item for sublist in docs for item in sublist]
# print(docs_list)
# print(f"len of documents :{len(docs_list)}")
# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=512, chunk_overlap=0
# )
# doc_splits = text_splitter.split_documents(docs_list)
# print(f"length of document chunks generated :{len(doc_splits)}")

# # Optimize chunk size and reduce memory usage
# doc_splits = doc_splits[:min(len(doc_splits), 100)]  # Limit number of chunks

# # Use lighter embedding model and reduced dimensions
# embed_model.max_seq_length = 128  # Reduce sequence length

# # Create vectorstore with optimized settings
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     embedding=embed_model,
#     collection_name="local-rag",
#     collection_metadata={"hnsw:space": "cosine", "hnsw:M": 8}  # Reduce graph complexity
# )

# # Configure retriever for faster retrieval
# retriever = vectorstore.as_retriever(search_kwargs={"k":2})


# # Import only what's needed
# from langchain.prompts import PromptTemplate 
# from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
# import time

# prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a
#     user question to a vectorstore or web search. Use the vectorstore for questions on LLM  agents,
#     prompt engineering,AI research insights,Blogs related and adversarial attacks.if asked bout latest or recent blogs on ai,agents,tools use vectorstore. You do not need to be stringent with the keywords
#     in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search'
#     or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and
#     no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#     input_variables=["question"],
# )
# start = time.time()
# question_router = prompt | llm | JsonOutputParser()
# #
# question = "latest blog on hugging face"
# print(question_router.invoke({"question": question}))
# end = time.time()
# print(f"The time required to generate response by Router Chain in seconds:{end - start}")
# prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
#     Use the following pieces of retrieved context to answer the question. If you don't know the answer, just give relavant answere and don't mention about the context.
#     Use five sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
#     Question: {question}
#     Context: {context}
#     Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#     input_variables=["question", "document"],
# )

# # Post-processing
# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# # Chain
# start = time.time()
# rag_chain = prompt | llm | StrOutputParser()
# end = time.time()
# print(f"The time required to generate response by Router Chain in seconds:{end - start}")


# prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
#     of a retrieved document to a user question. If the document contains keywords related to the user question,
#     grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
#     Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
#     Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
#      <|eot_id|><|start_header_id|>user<|end_header_id|>
#     Here is the retrieved document: \n\n {document} \n\n
#     Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
#     """,
#     input_variables=["question", "document"],
# )


# start = time.time()
# retrieval_grader = prompt | llm | JsonOutputParser()
# question = "latest ai blog"
# docs = retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": doc_txt}))

# # generation=retrieval_grader.invoke({"question": question, "document": doc_txt})
# end = time.time()
# print(f"The time required to generate response by the retrieval grader in seconds:{end - start}")


# # Prompt
# prompt = PromptTemplate(
#     template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether
#     an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate
#     whether the answer is grounded in / supported by a set of facts.Grade as 'yes' if ANY of these are true:
#     1. Answer contains specific details from documents
#     2. Answer directionally matches document themes
#     3. Documents mention related entities/context.
#     If the answer contains 'yes' or is related to blogs, latest news, or recent research updates, validate and return 'yes'. Provide the binary score as a JSON with a
#     single key 'score' and no preamble or explanation.<|eot_id|><|start_header_id|>user<|end_header_id|>
#     Here are the facts:
#     \n ------- \n
#     {documents}
#     \n ------- \n
#     Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#     input_variables=["generation", "documents"],
# )
# start = time.time()
# hallucination_grader = prompt | llm | JsonOutputParser()
# generation='yes'
# hallucination_grader_response = hallucination_grader.invoke({"documents": docs, "generation": generation})
# print(prompt.format(documents=docs, generation=generation))

# end = time.time()
# print(f"The time required to generate response by the generation chain in seconds:{end - start}")
# print(hallucination_grader_response)


# prompt = PromptTemplate(
#     template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an
#     answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is
#     useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.Always return yes
#      <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
#     \n ------- \n
#     {generation}
#     \n ------- \n
#     Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
#     input_variables=["generation", "question"],
# )
# start = time.time()
# answer_grader = prompt | llm | JsonOutputParser()
# answer_grader_response = answer_grader.invoke({"question": question,"generation": generation})
# end = time.time()
# print(f"The time required to generate response by the answer grader in seconds:{end - start}")
# print(answer_grader_response)
# web_search_tool = TavilySearchResults(api_wrapper=tavily_search_api_wrapper,k=3)
# from typing_extensions import TypedDict
# from typing import List,Optional
# from langchain.schema import Document
# ### State

# class GraphState(TypedDict):
#     question : str
#     generation : str
#     web_search : str
#     documents : Optional[List[Document]] # make documents optional

# from langchain.schema import Document
# def retrieve(state):
#     """
#     Retrieve documents from vectorstore

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, documents, that contains retrieved documents
#     """
#     print("---RETRIEVE---")
#     question = state["question"]

#     # Retrieval
#     documents = retriever.invoke(question)
#     return {"documents": documents, "question": question}
# def generate(state):
#     """
#     Generate answer using RAG on retrieved documents

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, generation, that contains LLM generation
#     """
#     print("---GENERATE---")
#     question = state["question"]
#     documents = state["documents"]

#     # RAG generation
#     generation = rag_chain.invoke({"context": documents, "question": question})
#     return {"documents": documents, "question": question, "generation": generation}
# def grade_documents(state):
#     """
#     Enhanced document relevance grading with parallel processing and threshold-based scoring
#     """
#     print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
#     question = state["question"]
#     documents = state["documents"]

#     # Batch process documents for efficiency
#     graded_docs = []
#     for d in documents:
#         try:
#             score = retrieval_grader.invoke({
#                 "question": question,
#                 "document": d.page_content[:10000]  # Handle long documents
#             })
#             graded_docs.append((d, score['score'].lower()))
#         except Exception as e:
#             print(f"Error grading document: {e}")
#             graded_docs.append((d, "no"))

#     # Calculate relevance threshold (at least 50% docs relevant)
#     relevant_count = sum(1 for _, score in graded_docs if score == "yes")
#     relevance_ratio = relevant_count / len(graded_docs) if len(graded_docs) > 0 else 0

#     # Determine web search need
#     web_search = "Yes" if relevance_ratio < 0.5 else "No"

#     # Filter documents with explanation
#     filtered_docs = [d for d, score in graded_docs if score == "yes"]
#     print(f"Relevance ratio: {relevance_ratio:.2f} ({relevant_count}/{len(graded_docs)})")

#     return {
#         "documents": filtered_docs,
#         "question": question,
#         "web_search": web_search,
#         "graded_docs": graded_docs  # Keep for debugging
#     }
# def web_search(state):
#     """
#     Web search based based on the question

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): Appended web results to documents
#     """

#     print("---WEB SEARCH---")
#     question = state["question"]
#     documents = state.get("documents", [])

#     # Web search
#     docs = web_search_tool.invoke({"query": question})
#     web_results = "\n".join([d["content"] for d in docs])
#     web_results = Document(page_content=web_results)
#     if documents is not None:
#         documents.append(web_results)
#     else:
#         documents = [web_results]
#     return {"documents": documents, "question": question}
# #
# def route_question(state):
#     """
#     Route question to web search or RAG.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Next node to call
#     """

#     print("---ROUTE QUESTION---")
#     question = state["question"]
#     print(question)
#     source = question_router.invoke({"question": question})
#     print(source)
#     print(source['datasource'])
#     if source['datasource'] == 'web_search':
#         print("---ROUTE QUESTION TO WEB SEARCH---")
#         return "websearch"
#     elif source['datasource'] == 'vectorstore':
#         print("---ROUTE QUESTION TO RAG---")
#         return "vectorstore"
    
# def decide_to_generate(state):
#     """
#     Determines whether to generate an answer, or add web search

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Binary decision for next node to call
#     """

#     print("---ASSESS GRADED DOCUMENTS---")
#     question = state["question"]
#     web_search = state["web_search"]
#     filtered_documents = state["documents"]

#     if web_search == "Yes":
#         # All documents have been filtered check_relevance
#         # We will re-generate a new query
#         print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
#         return "websearch"
#     else:
#         # We have relevant documents, so generate answer
#         print("---DECISION: GENERATE---")
#         return "generate"
# from pprint import pprint

# def grade_generation_v_documents_and_question(state):
#     """
#     Determines whether the generation is grounded in the document and answers question.

#     Args:
#         state (dict): The current graph state

#     Returns:
#         str: Decision for next node to call
#     """

#     print("---CHECK HALLUCINATIONS---")
#     question = state["question"]
#     documents = state["documents"]
#     generation = state["generation"]

#     score = hallucination_grader.invoke({"documents": documents, "generation": generation})
#     grade = score['score']

#     # Check hallucination
#     if grade == "yes":
#         print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
#         # Check question-answering
#         print("---GRADE GENERATION vs QUESTION---")
#         score = answer_grader.invoke({"question": question,"generation": generation})
#         grade = score['score']
#         if grade == "yes":
#             print("---DECISION: GENERATION ADDRESSES QUESTION---")
#             return "useful"
#         else:
#             print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
#             return "not useful"
#     else:
#         pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
#         return "not supported"
# from langgraph.graph import END, StateGraph
# workflow = StateGraph(GraphState)

# # Define the nodes
# workflow.add_node("websearch", web_search) # web search
# workflow.add_node("retrieve", retrieve) # retrieve
# workflow.add_node("grade_documents", grade_documents) # grade documents
# workflow.add_node("generate", generate) # generatae
# workflow.set_conditional_entry_point(
#     route_question,
#     {
#         "websearch": "websearch",
#         "vectorstore": "retrieve",
#     },
# )

# workflow.add_edge("retrieve", "grade_documents")
# workflow.add_conditional_edges(
#     "grade_documents",
#     decide_to_generate,
#     {
#         "websearch": "websearch",
#         "generate": "generate",
#     },
# )
# workflow.add_edge("websearch", "generate")
# workflow.add_conditional_edges(
#     "generate",
#     grade_generation_v_documents_and_question,
#     {
#         "not supported": END,
#         "useful": END,
#         "not useful": "websearch",
#     },
# )

# apps = workflow.compile()
# # from pprint import pprint
# # inputs = {"question": "What is the latest ai news?"}
# # for output in app.stream(inputs):
# #     for key, value in output.items():
# #         pprint(f"Finished running: {key}:")
# # pprint(value["generation"])

# # from IPython.display import Image

# # Image(app.get_graph().draw_png())