from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from src.data.vectorstore.chroma import get_vector_store
from src.llm.llm import get_llm


class RAGResponse(BaseModel):
    """Structured output for the RAG system."""

    answer: str = Field(
        description="The natural language answer to the user's question, referencing specific movies."
    )
    contexts: List[str] = Field(
        description="Exact snippets from the retrieved movie plots that support the answer."
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


def query_rag(query: str) -> Dict[str, Any]:
    """
    Full RAG pipeline: Retrieve -> Generate -> Structured Output.
    """
    docs = retrieve_relevant_chunks(query)

    context_snippets = [
        f"[{d.metadata.get('title', 'Unknown')}]: {d.page_content[:500]}..."
        if len(d.page_content) > 500
        else f"[{d.metadata.get('title', 'Unknown')}]: {d.page_content}"
        for d in docs
    ]

    context_text = "\n\n".join(
        [
            f"--- MOVIE: {d.metadata.get('title', 'Unknown')} ---\n{d.page_content}"
            for d in docs
        ]
    )

    llm = get_llm()
    structured_llm = llm.with_structured_output(RAGResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a movie expert assistant. Answer the user's question based ONLY on the provided context snippets.

Rules:
1. If a movie is mentioned by name in the question, look for that specific movie in the context.
2. The 'contexts' field MUST contain the relevant plot snippets you used to form your answer.
3. If the answer cannot be found in the context, say so clearly.
4. Always reference the movie title in your answer.""",
            ),
            ("user", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    chain = prompt | structured_llm
    response = chain.invoke({"context": context_text, "question": query})

    # Ensure contexts are populated from retrieved docs if LLM didn't include them
    result = response.model_dump()
    if not result["contexts"] and docs:
        result["contexts"] = context_snippets

    return result
