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
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    # FIX: Robust cleaning for broken PDFs
    import re

    md_text = re.sub(r'(\w)-\n(\w)', r'\1\2', md_text)
    md_text = re.sub(r'(\w)\n(\w)', r'\1\2', md_text)

    md_text = re.sub(
        r'https?://[^\s>\]]+(?:\n[^\s>\]]+)+',
        lambda m: m.group(0).replace('\n', ''),
        md_text
    )

    md_text = re.sub(r'\n{3,}', '\n\n', md_text)
    
    # Extract hyperlinks from PDF annotations
    import fitz
    doc = fitz.open(pdf_path)
    extracted_urls = set()

    for page in doc:
        for link in page.get_links():
            if 'uri' in link:
                extracted_urls.add(link['uri'])

    # ALSO extract URLs from text
    url_pattern = r'https?://[^\s\)\]]+'
    found_urls = re.findall(url_pattern, md_text)
    extracted_urls.update(found_urls)
                
    if extracted_urls:
        #  UPDATED: Anchor-based links
        md_text += "\n\n### Important Links\n"
        for url in extracted_urls:
            if "github.com" in url:
                md_text += f"- [Code Repository (GitHub)]({url})\n"
            else:
                md_text += f"- [Project Page]({url})\n"

        #  Labeled URLs for embedding strength
        md_text += "\n\n### Direct Links for Retrieval\n"
        for url in extracted_urls:
            if "github.com" in url:
                md_text += f"Code Repository Link: {url}\n"
            else:
                md_text += f"Project Page Link: {url}\n"

        #  Semantic reinforcement (CRITICAL)
        md_text += "\n\n### Link Descriptions\n"
        for url in extracted_urls:
            if "github.com" in url:
                md_text += f"This document contains the GitHub code repository link: {url}\n"
                md_text += f"The implementation code is available here: {url}\n"
            else:
                md_text += f"This document contains the official project page: {url}\n"
                md_text += f"More details about the project can be found here: {url}\n"

    file_name = os.path.basename(pdf_path)
    
    return [Document(page_content=md_text, metadata={"source": file_name})]


def get_text_splitter():
    return MarkdownTextSplitter(
        chunk_size=1500,
        chunk_overlap=250
    )


def process_pdfs(pdf_paths):
    all_chunks = []
    splitter = get_text_splitter()
    
    for path in pdf_paths:
        try:
            raw_docs = extract_pdf_content(path)
            chunks = splitter.split_documents(raw_docs)

            for chunk in chunks:
                chunk.metadata['source'] = raw_docs[0].metadata['source']

                text = chunk.page_content.lower()

                # METADATA BOOST + semantic injection
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