"""
Chat Route — Primary API Endpoint

Handles all chat requests and routes them to the correct
RAG pipeline based on the rag_type field in the request.

Supported pipelines:
    "kb"     → Phase 1 — Bedrock Knowledge Base RAG
    "custom" → Phase 2 — Custom OpenSearch RAG (default)

Request schema:  ChatRequest  (message, rag_type)
Response schema: ChatResponse (answer, context)
"""

import logging
from fastapi import APIRouter, HTTPException, Depends

from api.schemas.chat_schema import ChatRequest, ChatResponse
from api.dependencies import get_kb_rag_pipeline, get_custom_rag_pipeline
from rag.pipelines.kb_rag import KBRAGPipeline
from rag.pipelines.custom_rag import CustomRAGPipeline, CustomRAGResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    kb_pipeline: KBRAGPipeline = Depends(get_kb_rag_pipeline),
    custom_pipeline: CustomRAGPipeline = Depends(get_custom_rag_pipeline)
):
    """
    Main chat endpoint for the AtoZ AI Assistant.

    Routes the request to the correct RAG pipeline:
    - rag_type = "kb"     → Bedrock Knowledge Base (Phase 1)
    - rag_type = "custom" → Custom OpenSearch RAG  (Phase 2, default)

    Both pipelines return answer + context/sources.
    Response schema is identical regardless of pipeline used.
    """
    try:
        logger.info(
            f"Chat request received | "
            f"pipeline: {request.rag_type} | "
            f"query: '{request.message}'"
        )

        if request.rag_type == "kb":
            return await _handle_kb_pipeline(request.message, kb_pipeline)
        else:
            return await _handle_custom_pipeline(request.message, custom_pipeline)

    except ValueError as e:
        # Raised by pipelines for invalid input (empty query etc.)
        logger.warning(f"Invalid request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Chat endpoint failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Chat processing failed. Please try again."
        )


async def _handle_kb_pipeline(
    message: str,
    pipeline: KBRAGPipeline
) -> ChatResponse:
    """
    Run KB RAG pipeline and normalize response to ChatResponse.

    Phase 1 pipeline — Bedrock Knowledge Base retrieval.
    """
    logger.info("Routing to KB RAG pipeline")

    result = await pipeline.run(message)

    # KB pipeline returns dict with "answer" and "documents" keys
    return ChatResponse(
        answer=result["answer"],
        context=result.get("documents", [])
    )


async def _handle_custom_pipeline(
    message: str,
    pipeline: CustomRAGPipeline
) -> ChatResponse:
    """
    Run Custom RAG pipeline and normalize response to ChatResponse.

    Phase 2 pipeline — Custom OpenSearch + Titan Embeddings retrieval.
    """
    logger.info("Routing to Custom RAG pipeline")

    # CustomRAGPipeline.run() is async — must be awaited
    result: CustomRAGResponse = await pipeline.run(message)

    # result is a dataclass — use dot notation, not dict access
    return ChatResponse(
        answer=result.answer,
        context=result.sources
    )