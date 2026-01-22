import argparse
import sys
from src.data.loader import load_movie_plots
from src.data.chunker import chunk_movie_plots
from src.data.vectorstore.chroma import add_documents_to_store, get_vector_store
from src.data.rag.rag import query_rag


def ingest_data():
    """Lengths processed -> Chunked -> Stored in ChromaDB"""
    print("--- Starting Ingestion Process ---")

    print("Loading data...")
    df = load_movie_plots(n_rows=300)
    print(f"Loaded {len(df)} movie plots.")

    print("Chunking data...")
    documents = chunk_movie_plots(df)
    print(f"Created {len(documents)} text chunks.")

    print("Embedding and Storing...")
    add_documents_to_store(documents)
    print("--- Ingestion Complete ---")


def run_query_loop():
    """Interactive loop for asking questions"""
    print("\n--- Movie RAG System Ready ---")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question about a movie plot: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        try:
            print("\nThinking...")
            result = query_rag(query)

            import json

            print(json.dumps(result, indent=2))
            print("-" * 50)
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Movie Plot RAG System")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Load and index movie data before querying",
    )
    args = parser.parse_args()

    if args.ingest:
        ingest_data()

    try:
        get_vector_store()

    except Exception as e:
        print(
            f"Could not connect to Vector Store. Make sure Docker is running. Error: {e}"
        )
        sys.exit(1)

    run_query_loop()


if __name__ == "__main__":
    main()
