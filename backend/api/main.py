from fastapi import FastAPI
from .query_retriever_multiple import router as query_retriever_multiple_router

app = FastAPI()
app.include_router(query_retriever_multiple_router)
