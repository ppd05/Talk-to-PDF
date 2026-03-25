import os
import re
import fitz
import pymupdf4llm
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from vector_store import add_documents_to_db

def extract_pdf_content(pdf_path):
    """
    Extracts text dynamically using pymupdf4llm for superior Markdown rendering.
    If ONNX OCR encounters a tensor crash on unsupported image data in complex PDFs,
    it natively falls back to standard Langchain PyPDFLoader to guarantee extraction!
    """
    file_name = os.path.basename(pdf_path)
    try:
        # ATTEMPT 1: Strict Markdown + Hyperlinks via PyMuPDF4LLM
        md_text = pymupdf4llm.to_markdown(pdf_path, write_images=False)
        
        # FIX: Robust cleaning for broken PDFs
        md_text = re.sub(r'(\w)-\n(\w)', r'\1\2', md_text)
        md_text = re.sub(r'(\w)\n(\w)', r'\1\2', md_text)
        md_text = re.sub(
            r'https?://[^\s>\]]+(?:\n[^\s>\]]+)+',
            lambda m: m.group(0).replace('\n', ''),
            md_text
        )
        md_text = re.sub(r'\n{3,}', '\n\n', md_text)
        
        # EXTRACT: Annotations
        doc = fitz.open(pdf_path)
        extracted_urls = set()
        for page in doc:
            for link in page.get_links():
                if 'uri' in link:
                    extracted_urls.add(link['uri'])

        url_pattern = r'https?://[^\s\)\]]+'
        found_urls = re.findall(url_pattern, md_text)
        extracted_urls.update(found_urls)
                    
        if extracted_urls:
            md_text += "\n\n### Important Links\n"
            for url in extracted_urls:
                if "github.com" in url:
                    md_text += f"- [Code Repository (GitHub)]({url})\n"
                else:
                    md_text += f"- [Project Page]({url})\n"
            
            md_text += "\n\n### Direct Links for Retrieval\n"
            for url in extracted_urls:
                if "github.com" in url:
                    md_text += f"Code Repository Link: {url}\n"
                else:
                    md_text += f"Project Page Link: {url}\n"
            
            md_text += "\n\n### Link Descriptions\n"
            for url in extracted_urls:
                if "github.com" in url:
                    md_text += f"This document contains the GitHub code repository link: {url}\n"
                    md_text += f"The implementation code is available here: {url}\n"
                else:
                    md_text += f"This document contains the official project page: {url}\n"
                    md_text += f"More details about the project can be found here: {url}\n"
                    
        return [Document(page_content=md_text, metadata={"source": file_name})]
        
    except Exception as e:
        print(f"[!] PyMuPDF4LLM extraction failed on {file_name}: {e}\n[+] Triggering PyPDFLoader Fallback!")
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            full_text = "\n\n".join(doc.page_content for doc in docs)
            return [Document(page_content=full_text, metadata={"source": file_name})]
        except Exception as fallback_e:
            raise Exception(f"Ultimate Fallback Extraction Failed: {fallback_e}")

def get_text_splitter():
    """Returns the text splitter config."""
    # Using RecursiveCharacterTextSplitter for robust code preservation
    return RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=300,
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

            for chunk in chunks:
                chunk.metadata['source'] = raw_docs[0].metadata['source']
                text = chunk.page_content.lower()

                # Semantic reinforcement
                if "github.com" in text:
                    chunk.metadata["has_code_link"] = True
                    chunk.page_content += "\nThis chunk contains the GitHub code repository link."
                if "project page" in text or "project" in text:
                    chunk.metadata["has_project_link"] = True
                    chunk.page_content += "\nThis chunk contains the project page link."

            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    if all_chunks:
        add_documents_to_db(all_chunks)
        return len(all_chunks)
    
    return 0