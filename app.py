import streamlit as st
import os
from ingestion import process_pdfs
from rag_pipeline import answer_query
from dotenv import load_dotenv

# Set up page configurations for Streamlit
st.set_page_config(
    page_title="PDF Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme via Streamlit config isn't programmable inside script,
# But setting page config helps. User can set theme in .streamlit/config.toml

# Ensure data directories exist
os.makedirs("data/raw_pdfs", exist_ok=True)

def main():
    st.title("📚 Intelligent PDF Q&A System")
    st.markdown("Upload multiple PDFs and ask questions based ONLY on their content.")

    # Sidebar for uploading and processing documents
    with st.sidebar:
        st.header("📂 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type=["pdf"], 
            accept_multiple_files=True
        )
        
        if st.button("Process Documents", type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing documents... This might take a while."):
                    saved_paths = []
                    # Save PDFs to local data directory
                    for u_file in uploaded_files:
                        file_path = os.path.join("data/raw_pdfs", u_file.name)
                        with open(file_path, "wb") as f:
                            f.write(u_file.read())
                        saved_paths.append(file_path)
                    
                    try:
                        # Process PDFs: extract, chunk, and embed
                        num_chunks = process_pdfs(saved_paths)
                        if num_chunks > 0:
                            st.success(f"Successfully processed and indexed {len(saved_paths)} PDFs into {num_chunks} chunks!")
                        else:
                            st.warning("No new text found or extraction failed.")
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")

    st.divider()

    # Main area for Query Input and AI Answer
    st.header("💬 Ask Question")
    
    query = st.text_input("Enter your query about the uploaded documents:")

    if query:
        st.subheader("🔍 User Query")
        st.info(query)

        with st.spinner("Searching for answers..."):
            answer, source_docs = answer_query(query, return_source_docs=True)

        st.subheader("🤖 AI Answer")
        st.markdown(answer)
        
        with st.expander("View Source Documents Context"):
            if isinstance(source_docs, list) and len(source_docs) > 0:
                for i, doc in enumerate(source_docs):
                    source_file = doc.metadata.get("source", "Unknown")
                    page_num = doc.metadata.get("page", "Unknown")
                    st.markdown(f"**Source {i+1}** - File: `{source_file}`, Page: `{page_num}`")
                    st.text(doc.page_content)
                    st.divider()
            else:
                st.write("No source documents found or an error occurred.")

if __name__ == '__main__':
    # Load environment variables
    load_dotenv()
    
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API Key not found! Please check your .env file.")
        st.stop()
        
    main()
