import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_embeddings():
    """Returns the Google Gemini Embedding Model."""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
