import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from vector_store import setup_vector_store
from dotenv import load_dotenv

load_dotenv()

def format_docs(docs):
    """Format documents into a single string for context."""
    return "\n\n".join(doc.page_content for doc in docs)

def get_rag_chain():
    """Setup and return the RAG pipeline chain."""
    vectorstore = setup_vector_store()
    
    # Retrieve top 8 chunks to broaden the search context slightly
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    # Initialize Gemini model (Using the model ID provided in requirements)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview", # Fallback/closest valid API model name if gemini-flash-lite-25 fails
        temperature=0,
    )

    template = """Answer ONLY from context. If not found, say 'Not in document'.
    If the context contains a table, you MUST extract and output it as a proper Markdown table (using `|` columns and `---` header separators). 
    Do NOT flatten tables into plain text sentences.
    If the context contains a code block, preserve its formatting exactly using triple backticks.
    If the context contains a link (like github or project page), return it as a FULL, CLICKABLE markdown link.
    IMPORTANT: If a URL or link is broken or split across multiple lines in the context, carefully concatenate the pieces into a SINGLE unbroken URL before rendering it.

    Context:
    {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    # Construct the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def answer_query(query: str, return_source_docs: bool = False):
    """Answer a user query using the RAG pipeline."""
    try:
        if return_source_docs:
            vectorstore = setup_vector_store()
            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})
            docs = retriever.invoke(query)
            
            chain = get_rag_chain()
            answer = chain.invoke(query)
            return answer, docs
        else:
            chain = get_rag_chain()
            return chain.invoke(query)
            
    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        if return_source_docs:
            return error_msg, []
        return error_msg
