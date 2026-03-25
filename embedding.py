import os
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """Returns free Hugging Face Embeddings."""
    # Using a fast, lightweight, and very effective model for semantic search
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Default to CPU to ensure it runs everywhere
        encode_kwargs={'normalize_embeddings': True}
    )
