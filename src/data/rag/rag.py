from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from src.data.vectorstore.chroma import get_vector_store
from src.llm.llm import get_llm


class RAGResponse(BaseModel):
    """Structured output for the RAG system."""

    answer: str = Field(
        description="A concise, natural language answer that directly addresses the question. "
        "Include the movie title (*Movie Title*) and summarize the relevant plot details."
    )
    contexts: List[str] = Field(
        description="The exact plot snippets from retrieved documents that were used to form the answer."
    )
    reasoning: str = Field(
        description="A brief explanation of the search and reasoning process: "
        "what the question asked for, what was found, and how the answer was formed."
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
        f"{d.metadata.get('title', 'Unknown')}: {d.page_content[:300]}..."
        if len(d.page_content) > 300
        else f"{d.metadata.get('title', 'Unknown')}: {d.page_content}"
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
                """You are a movie expert assistant. Answer questions using ONLY the provided context.

Output Format:
- **answer**: A concise summary that directly answers the question. Use *italics* for movie titles. Combine relevant plot details into a clear, informative response.
- **contexts**: Include the specific plot excerpts you referenced (can be shortened).
- **reasoning**: Briefly explain: (1) what the user asked, (2) what you found in the context, (3) how you formed the answer.

Guidelines:
- If the user asks about a specific movie, focus on that movie's plot.
- If the user asks a general question (e.g., "movies about AI"), find relevant movies and summarize.
- If the information isn't in the context, clearly state that.
- Keep answers informative but concise.""",
            ),
            ("user", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )

    chain = prompt | structured_llm
    response = chain.invoke({"context": context_text, "question": query})

    result = response.model_dump()
    if not result["contexts"] and docs:
        result["contexts"] = context_snippets

    return result
