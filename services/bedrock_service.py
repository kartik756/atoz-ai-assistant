"""
This service interacts with AWS Bedrock.

Responsibilities:
1. Retrieve documents from Bedrock Knowledge Base  (Phase 1)
2. Generate answers using Bedrock LLM              (Phase 1)
3. Generate vector embeddings using Titan          (Phase 2)
"""

import boto3
import logging
import json
from typing import List

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BedrockService:
    """
    Central service for all AWS Bedrock interactions.

    Phase 1:
        - retrieve_from_kb()    → fetch relevant docs from Bedrock KB
        - generate_response()   → generate answer using Nova Lite LLM

    Phase 2:
        - embed_text()          → convert text to vector using Titan Embeddings
    """

    def __init__(self):

        self.region = settings.AWS_REGION
        self.kb_id = settings.BEDROCK_KNOWLEDGE_BASE_ID
        self.model_id = settings.BEDROCK_MODEL_ID
        self.embedding_model_id = settings.EMBEDDING_MODEL_ID  # added Phase 2

        # Client for LLM inference + embeddings
        self.bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=self.region
        )

        # Client for Knowledge Base retrieval
        self.bedrock_agent_runtime = boto3.client(
            "bedrock-agent-runtime",
            region_name=self.region
        )

    # ──────────────────────────────────────────────
    # Phase 1 Methods (unchanged)
    # ──────────────────────────────────────────────

    async def retrieve_from_kb(self, query: str) -> List[str]:
        """
        Retrieve relevant document chunks from Bedrock Knowledge Base.
        Used by: rag/pipelines/kb_rag.py
        """
        try:
            logger.info(f"Retrieving documents from KB for query: {query}")

            response = self.bedrock_agent_runtime.retrieve(
                knowledgeBaseId=self.kb_id,
                retrievalQuery={"text": query}
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
        Generate an answer using Bedrock LLM with retrieved context.
        Used by: rag/pipelines/kb_rag.py  AND  rag/pipelines/custom_rag.py
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
- If the answer is not found in the context, say you don't know
- Keep the answer concise and clear
"""

            response = self.bedrock_runtime.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
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

    # ──────────────────────────────────────────────
    # Phase 2 Methods
    # ──────────────────────────────────────────────

    async def embed_text(self, text: str) -> List[float]:
        """
        Convert text into a vector embedding using Amazon Titan Embeddings v2.

        Used by:
            - rag/ingestion/embeddings.py  → embed document chunks at ingestion time
            - rag/retrieval/retriever.py   → embed user query at search time

        Args:
            text: Raw text string to embed (chunk or query)

        Returns:
            List of 1536 floats representing the semantic meaning of the text
        """
        try:
            logger.info(f"Generating embedding for text of length {len(text)}")

            # Titan Embeddings expects this exact request body shape
            body = json.dumps({
                "inputText": text,
               # "dimensions": settings.EMBEDDING_DIMENSION,   # 1536
                #"normalize": True    # normalizes vector to unit length
                                     # required for cosine similarity in OpenSearch
            })

            response = self.bedrock_runtime.invoke_model(
                modelId=self.embedding_model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )

            response_body = json.loads(response["body"].read())
            embedding = response_body["embedding"]

            logger.info(f"Embedding generated: {len(embedding)} dimensions")
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise