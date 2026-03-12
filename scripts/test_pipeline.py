from rag.pipelines.custom_rag import CustomRAGPipeline


def test_rag():
    """
    Test RAG pipeline with sample query.
    """

    rag = CustomRAGPipeline()

    question = "What is the leave policy?"

    result = rag.run(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(
            f'{source["document_name"]} - page {source["page_number"]}'
        )


if __name__ == "__main__":
    test_rag()