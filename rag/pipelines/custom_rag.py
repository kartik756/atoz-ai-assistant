from rag.retrieval.retriever import Retriever
from services.bedrock_service import BedrockService


class CustomRAGPipeline:
    """
    Custom RAG pipeline implementation.

    This pipeline:
    1. Retrieves relevant document chunks
    2. Builds context from retrieved chunks
    3. Sends context + question to LLM
    4. Returns generated answer
    """

    def __init__(self):
        self.retriever = Retriever()
        self.bedrock_service = BedrockService()

    def run(self, question: str):
        """
        Execute RAG pipeline for a given user question.
        """

        # Step 1: Retrieve relevant document chunks
        retrieved_docs = self.retriever.retrieve(question)

        # Step 2: Build context string
        context = self._build_context(retrieved_docs)

        # Step 3: Build prompt for LLM
        prompt = self._build_prompt(question, context)

        # Step 4: Generate answer using Bedrock
        response = self.bedrock_service.generate_response(prompt)

        return {
            "answer": response,
            "sources": retrieved_docs
        }

    def _build_context(self, documents: list):
        """
        Combine retrieved documents into a context string.
        """

        context_parts = []

        for doc in documents:
            context_parts.append(doc["text"])

        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str):
        """
        Build final prompt sent to the LLM.
        """

        prompt = f"""
You are an internal HR assistant for the Amazon AtoZ portal.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
"""

        return prompt