from typing import List
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from src.data.vectorstore.chroma import get_vector_store


class RAGResponse(BaseModel):
    """Structured output for the RAG system."""

    answer: str = Field(
        description="The natural language answer to the user's question"
    )
    contexts: List[str] = Field(
        description="Specific snippets from the retrieved movie plots that support the answer"
    )
    reasoning: str = Field(
        description="A brief explanation of how the retrieved context was used to derive the answer"
    )


def retrieve_relevant_documents(question: str, k: int = 5) -> List[Document]:
    """Retrieve relevant documents from the vector store based on the user's question.

    Args:
        question: The user's input question.
        k: Number of top documents to retrieve.

    Returns:
        List of retrieved Document objects.
    """
    vector_store = get_vector_store()
    return vector_store.similarity_search(query=question, k=k)
