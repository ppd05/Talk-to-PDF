import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import add_documents_to_db

def extract_pdf_content(pdf_path):
    """
    Extracts text from a given PDF using Langchain's PyPDFLoader.
    Returns a list of Document objects with metadata already intact.
    """
    loader = PyPDFLoader(pdf_path)
    # PyPDFLoader automatically loads and splits by pages, 
    # preserving source and page number in the metadata.
    documents = loader.load()
    return documents

def get_text_splitter():
    """Returns the text splitter config."""
    return RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )

def process_pdfs(pdf_paths):
    """
    Given a list of PDF file paths, extracts content, chunks it, and adds it to the vector DB.
    """
    all_chunks = []
    splitter = get_text_splitter()
    
    for path in pdf_paths:
        try:
            raw_docs = extract_pdf_content(path)
            chunks = splitter.split_documents(raw_docs)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    if all_chunks:
        add_documents_to_db(all_chunks)
        return len(all_chunks)
    
    return 0
