from fastapi import APIRouter, HTTPException
from rag.pipelines.kb_rag import KBRAGPipeline
from api.schemas.chat_schema import ChatRequest, ChatResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize RAG pipeline
kb_rag_pipeline = KBRAGPipeline()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for employee assistant
    """

    try:

        logger.info(f"Received chat query: {request.message}")

        # Run RAG pipeline
        result = await kb_rag_pipeline.run(request.message)

        return ChatResponse(
            answer=result["answer"],
            context=result["documents"]
        )

    except Exception as e:

        logger.error(f"Chat endpoint failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Chat processing failed"
        )