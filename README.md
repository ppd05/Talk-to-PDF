# PDF RAG Question Answering System

This project is a complete, production-ready Full-Stack PDF Question Answering System using RAG (Retrieval-Augmented Generation). 
It leverages Google Gemini, Streamlit, Langchain, and ChromaDB.

## Features
- **Multi-PDF Upload**: Upload multiple PDF files simultaneously.
- **Smart Data Extraction**: Extracts normal text and tables (using `pdfplumber`).
- **RAG Pipeline**: Retrieves the top relevant document chunks using Google Generative AI Embeddings.
- **Persistent Vector DB**: Stores document embeddings inside a persistent ChromaDB instance.
- **Answer Quality**: Preserves table formats and provides highly accurate, context-bound answers.

## Architecture
- `app.py`: The Streamlit frontend and UI.
- `ingestion.py`: Handles loading PDFs, extracting text, tables, and chunking.
- `embedding.py`: Handles Google Generative AI embeddings generation.
- `vector_store.py`: Interacts with ChromaDB to store and retrieve chunks.
- `rag_pipeline.py`: Defines the LangChain retrieval and generation pipeline.

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <your_repo_url>
   cd interview
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

5. **Run the Application**
   ```bash
   streamlit run app.py
   ```
   *Note: For a forced dark theme, you can navigate to `.streamlit/config.toml` (create it if not present) and add:*
   ```toml
   [theme]
   base="dark"
   ```

## Git Integration (Pushing to GitHub)

Execute these commands in your terminal:
```bash
git init
git add .
git commit -m "Initial commit for PDF RAG System"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo-name.git
git push -u origin main
```
