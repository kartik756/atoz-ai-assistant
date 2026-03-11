from pydantic import BaseModel
from typing import List


class ChatRequest(BaseModel):
    """
    Request model for chat endpoint
    """

    message: str


class ChatResponse(BaseModel):
    """
    Response model returned by chat endpoint
    """

    answer: str
    context: List[str]