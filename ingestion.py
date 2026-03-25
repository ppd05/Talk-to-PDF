import os
import pymupdf4llm
from langchain.schema import Document
from langchain.text_splitter import MarkdownTextSplitter
from vector_store import add_documents_to_db

def extract_pdf_content(pdf_path):
    """
    Extracts text, tables, code blocks, and hyperlinks from a given PDF 
    using pymupdf4llm. It converts the entire PDF structure directly to Markdown!
    Returns a list containing a single Document with the entire markdown.
    """
    # to_markdown automatically reads the pdf and outputs strict markdown 
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    # Fix fragmented URLs natively before chunking! 
    import re
    md_text = re.sub(r'(https?://[^\s>]+)\n([^\s>]+)', r'\1\2', md_text)
    md_text = re.sub(r'(https?://[^\s>]+)\n([^\s>]+)', r'\1\2', md_text)
    
    # NEW FIX: Brute-force extract EVERY embedded clickable hyperlink directly from the PDF's internal syntax
    # Since some visible text differs from the actual clickable link, this grabs the hidden URLs natively!
    import fitz
    doc = fitz.open(pdf_path)
    extracted_urls = set()
    for page in doc:
        for link in page.get_links():
            if 'uri' in link:
                extracted_urls.add(link['uri'])
                
    if extracted_urls:
        md_text += "\n\n### Embedded Clickable Document Links (Raw Annotations)\n"
        for url in extracted_urls:
            md_text += f"- [{url}]({url})\n"

    file_name = os.path.basename(pdf_path)
    
    return [Document(page_content=md_text, metadata={"source": file_name})]

def get_text_splitter():
    """Returns the text splitter config tailored for Markdown structure."""
    # We use a larger chunk size to ensure tables and code snippets 
    # aren't randomly bisected during chunking!
    return MarkdownTextSplitter(
        chunk_size=1500,
        chunk_overlap=250
    )

def process_pdfs(pdf_paths):
    """
    Given a list of PDF file paths, extracts content, chunks it naturally using markdown headers, 
    and adds it to the vector DB.
    """
    all_chunks = []
    splitter = get_text_splitter()
    
    for path in pdf_paths:
        try:
            raw_docs = extract_pdf_content(path)
            chunks = splitter.split_documents(raw_docs)
            for chunk in chunks:
                # To help track source chunks explicitly per markdown segment natively.
                chunk.metadata['source'] = raw_docs[0].metadata['source']
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            
    if all_chunks:
        add_documents_to_db(all_chunks)
        return len(all_chunks)
    
    return 0
