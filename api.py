# from fastapi import FastAPI
# from pydantic import BaseModel
# from rag_pipeline import apps  # Import your RAG pipeline
# import uvicorn

# app = FastAPI(
#     title="RAG API",
#     description="API for querying RAG pipeline",
#     docs_url="/docs",  # Enable Swagger UI
#     redoc_url="/redoc"  # Enable alternative Redoc UI
# )

# class QueryRequest(BaseModel):
#     query: str

# @app.get("/")
# async def root():
#     return {"message": "API is running. Visit /docs for API documentation."}

# @app.post("/query", summary="Query RAG Pipeline")
# async def query_rag(request: QueryRequest):
#     output = apps.invoke({"query": request.query})
#     return {"response": output["response"]}

# if __name__ == "__main__":
#     uvicorn.run(app, port=8000)
# # host="0.0.0.0"
