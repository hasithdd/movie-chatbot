from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Returns the HuggingFace embedding model.
    Using 'all-MiniLM-L6-v2' which maps sentences to a 384 dimensional dense vector space.
    """
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
