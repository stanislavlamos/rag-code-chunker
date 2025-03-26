from fastapi import APIRouter, Request
from backend.pipeline import run_query_retriever_multiple


router = APIRouter()

@router.post("/query-retriever-multiple")
async def query_retriever_multiple(request: Request) -> dict[str, list[str]]:
    """
    Retriever for multiple queries
    :param request: request from the query and config info
    :return: retrieved answers together with the computed metrics
    """
    data = await request.json()
    answer = ["ahoj", "cau"]#run_query_retriever_multiple(data)
    return {"answer": answer}
