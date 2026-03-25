import os
from langchain_community.vectorstores import Chroma
from embedding import get_embeddings

VECTOR_STORE_DIR = "data/vector_db"

def setup_vector_store():
    """Initializes and returns the Chroma vector store."""
    if not os.path.exists(VECTOR_STORE_DIR):
        os.makedirs(VECTOR_STORE_DIR)
        
    embeddings = get_embeddings()
    
    vectorstore = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings,
        collection_name="pdf_rag_collection"
    )
    
    return vectorstore

def add_documents_to_db(chunks):
    """Adds chunked documents to the persistent ChromaDB."""
    vectorstore = setup_vector_store()
    vectorstore.add_documents(chunks)
    # Chroma automatically persists in newer versions, 
    # but explicit persistence might be required for older ones.
    if hasattr(vectorstore, 'persist'):
        vectorstore.persist()
