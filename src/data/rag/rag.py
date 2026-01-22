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


def retrieve_relevant_chunks(query: str, k: int = 5) -> List[Document]:
    """
    Retrieves the top-k relevant chunks from the vector store based on the query.

    Args:
        query: The user's question or input string.
        k: Number of top relevant chunks to retrieve.
    Returns:
        List of Document objects representing the most relevant chunks.
    """
    vector_store = get_vector_store()
    relevant_docs = vector_store.similarity_search(query, k=k)
    return relevant_docs
