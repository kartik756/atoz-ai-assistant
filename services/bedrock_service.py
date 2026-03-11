"""
This service interacts with AWS Bedrock.

Responsibilities:
1. Retrieve documents from Bedrock Knowledge Base
2. Generate answers using Bedrock LLM
"""

import boto3
import logging
from typing import List
import json

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BedrockService:
    """
    Service responsible for interacting with AWS Bedrock.

    Responsibilities:
    - Retrieve documents from Bedrock Knowledge Base
    - Generate responses using Bedrock models
    """

    def __init__(self):

        self.region = settings.AWS_REGION
        self.kb_id = settings.BEDROCK_KNOWLEDGE_BASE_ID
        self.model_id = settings.BEDROCK_MODEL_ID

        # Client for LLM inference
        self.bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=self.region
        )

        # Client for Knowledge Base retrieval
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            region_name=self.region
        )

    async def retrieve_from_kb(self, query: str) -> List[str]:
        """
        Retrieve relevant document chunks from Bedrock Knowledge Base
        """

        try:

            logger.info(f"Retrieving documents from KB for query: {query}")

            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={
                    "text": query
                }
            )

            results = response.get("retrievalResults", [])

            documents: List[str] = []

            for item in results:
                content = item.get("content", {})
                text = content.get("text")

                if text:
                    documents.append(text)

            logger.info(f"Retrieved {len(documents)} documents from KB")

            return documents

        except Exception as e:

            logger.error(f"KB retrieval failed: {str(e)}")
            raise

    async def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate an answer using Bedrock LLM with retrieved context
        """

        try:

            logger.info("Generating response using Bedrock model")

            context_text = "\n\n".join(context)

            prompt = f"""
You are an internal enterprise AI assistant for employees.

Use the following company documents to answer the question.

Context:
{context_text}

Question:
{query}

Instructions:
- Answer only using the provided context
- If the answer is not found in the context say you don't know
- Keep the answer concise and clear
"""

            response = self.bedrock_runtime.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt}
                        ]
                    }
                ],
                inferenceConfig={
                    "maxTokens": 500,
                    "temperature": 0.3
                }
            )

            answer = response["output"]["message"]["content"][0]["text"]

            logger.info("Bedrock model response generated")

            return answer

        except Exception as e:

            logger.error(f"Model generation failed: {str(e)}")
            raise
        