from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):

    message: str
    rag_type: Optional[str] = "custom"


class ChatResponse(BaseModel):

    answer: str
    context: Optional[List] = None