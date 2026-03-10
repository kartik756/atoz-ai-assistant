from typing import Dict, Any
from services.bedrock_service import BedrockService
from logging import get_logger

logger = get_logger(__name__)


class KBRAGPipeline:

    def __init__(self):
        self.bedrock_service = BedrockService()

    async def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the Knowledge Base RAG pipeline
        """

        try:

            logger.info(f"Starting KB RAG pipeline for query: {query}")

            # Step 1: Retrieve documents
            documents = await self.bedrock_service.retrieve_from_kb(query)

            # Step 2: Generate response
            answer = await self.bedrock_service.generate_response(
                query=query,
                context=documents
            )

            logger.info("KB RAG pipeline completed")

            return {
                "query": query,
                "answer": answer,
                "documents": documents
            }

        except Exception as e:

            logger.error(f"KB RAG pipeline failed: {str(e)}")

            raise