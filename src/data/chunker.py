from typing import List
import polars as pl
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def chunk_movie_plots(
    df: pl.DataFrame, chunk_size: int = 1800, chunk_overlap: int = 200
) -> List[Document]:
    """
    Chunks plot texts from the dataframe into smaller segments.

    Args:
        df: Polars DataFrame containing 'Title' and 'Plot' columns.
        chunk_size: Character limit for each chunk. 1800 chars is roughly 300 words.
        chunk_overlap: Overlap between chunks to maintain context.

    Returns:
        List of LangChain Document objects containing content and metadata.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = []

    for row in df.iter_rows(named=True):
        title = row["Title"]
        plot = row["Plot"]

        plot_chunks = text_splitter.create_documents(
            texts=[plot], metadatas=[{"title": title}]
        )
        documents.extend(plot_chunks)

    return documents
