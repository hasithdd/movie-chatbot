# filepath: /home/hasith/Personal/movie-rag/src/data/vectorstore/chroma.py
import chromadb
import uuid
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.data.embeddings.sentence_transformer import get_embedding_function

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "movie_plots"


def get_vector_store() -> Chroma:
    """
    Connects to the ChromaDB Docker instance and returns a LangChain vector store wrapper.
    """
    # Initialize the HTTP client to connect to the running Docker container
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    embedding_function = get_embedding_function()

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    return vector_store


def add_documents_to_store(documents: List[Document]) -> None:
    """
    Adds a list of Documents to the Chroma vector store.
    """
    if not documents:
        print("No documents to add.")
        return

    vector_store = get_vector_store()

    # Create unique IDs for each chunk to ensure idempotent additions
    ids = [str(uuid.uuid4()) for _ in documents]

    print(
        f"Adding {len(documents)} documents to ChromaDB collection '{COLLECTION_NAME}'..."
    )
    vector_store.add_documents(documents=documents, ids=ids)
    print("Documents added successfully.")
