from fastapi import APIRouter, HTTPException
from rag.pipelines.kb_rag import KBRAGPipeline
from rag.pipelines.custom_rag import CustomRAGPipeline
from api.schemas.chat_schema import ChatRequest, ChatResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize both pipelines
kb_rag_pipeline = KBRAGPipeline()
custom_rag_pipeline = CustomRAGPipeline()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint for employee assistant.
    Supports both Knowledge Base RAG and Custom RAG.
    """

    try:

        logger.info(f"Received chat query: {request.message}")

        # Select pipeline
        if request.rag_type == "kb":

            logger.info("Using Knowledge Base RAG pipeline")

            result = await kb_rag_pipeline.run(request.message)

            context = result["documents"]

        else:

            logger.info("Using Custom RAG pipeline")

            result = custom_rag_pipeline.run(request.message)

            context = result["sources"]

        return ChatResponse(
            answer=result["answer"],
            context=context
        )

    except Exception as e:

        logger.error(f"Chat endpoint failed: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Chat processing failed"
        )