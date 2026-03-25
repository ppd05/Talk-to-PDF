# Advanced Production-Ready RAG System for Complex PDFs

This project is a highly engineered, production-ready Full-Stack PDF Question Answering System using RAG (Retrieval-Augmented Generation). It allows users to upload complicated research papers, robustly extracting tables, code blocks, and hidden hyperlinks, and accurately retrieving answers using Google's Gemini models and Hugging Face Embeddings.

---

## 🚀 Advanced Technical Implementations & Architecture

To solve the most difficult challenges in modern generic RAG parsers, this repository features a completely custom **Semantic Reinforcement & Ingestion Pipeline**. 

### 1. Robust PDF Ingestion (`ingestion.py`)
Relying on generic PDF loaders often scrambles multi-column tables and visually hard-wraps URLs. We developed a proprietary pipeline utilizing **`pymupdf4llm`**:
- **Native Markdown Extraction**: Converts the entire PDF directly to strict Markdown structure (`.md`).
- **Table Integrity**: Visually drawn tables are algorithmically rendered into perfect Markdown grid syntax (e.g., `| Column 1 | Column 2 |`), making complex benchmarks 100% understandable to the LLM.

#### Multi-Tier Broken Link Recovery
Because research papers often physically chunk URLs across lines (e.g., `https://github... \n /project`), traditional Vector DBs fail to match user queries with broken links. We created a **Regex Auto-Healer**:
1. Seamlessly stitches words split by hyphenation (`-\n`).
2. Targets and concatenates fragmented `http` strings dynamically using regex lambda operations (`lambda m: m.group(0).replace('\n', '')`).

#### Semantic Reinforcement & Metadata Boosting (CRITICAL)
Instead of relying purely on mathematical embedding distances, we implemented highly advanced *Retrieval Reinforcement*:
- **Annotation & Deep Validation**: Uses raw `fitz` object extraction to grab hidden destination URLs from the PDF's click-annotation dictionary, cross-verifying them alongside physical text regex matches (`https?://[^\s)\]]+`).
- **Context Injection**: All valid URLs are forcefully injected at the bottom of the document block dynamically in three distinct ways to ensure the LLM *always* understands context:
   1. *Anchor Forms*: `- [Code Repository (GitHub)](URL)`.
   2. *Direct References*: `Code Repository Link: URL`.
   3. *Semantic Descriptions*: `This document contains the GitHub code repository link: URL`.
- **Chunk-Level Boosting**: Analyzes individual chunks before they hit the database. If a chunk mentions "github.com" or "project page", the Engine natively appends reinforcement text (`"This chunk contains the project page link"`) directly to the chunk and flags the vector's `metadata` with boolean boosts (`has_code_link=True`).

### 2. The Vector Database Engine (`embedding.py` & `vector_store.py`)
- **Smart Chunking**: Split via LangChain's `MarkdownTextSplitter` (Size: `1500`, Overlap: `250`). This deliberately respects the native Markdown boundaries so a table or code block isn't accidentally severed in half!
- **Hugging Face CPU Embeddings**: Utilizes the lightning-fast, highly contextual `sentence-transformers/all-MiniLM-L6-v2` model running natively on your system—no closed-source API payload costs.
- **Persistent Local DB**: Embedded vectors are saved persistently via a local `ChromaDB` instance (`data/vector_db/`), allowing for instant reloading.

### 3. Google Gemini Orchestration (`rag_pipeline.py`)
- Tied together via **LCEL** (LangChain Expression Language).
- Expanded Context Retrieval: Extracts the top `8` most relevant chunks (`k=8`).
- Uses `ChatGoogleGenerativeAI` targeting `gemini-3.1-flash-lite-preview` for generation.
- **Prompt Engineering Constraints**: Features an aggressive system prompt that strictly commands the LLM to preserve triple-backtick (`` ``` ``) code formatting and output strict `| --- |` Markdown tables rather than flattening retrieved data sets.

### 4. Interactive Frontend (`app.py`)
- Implemented with **Streamlit** using a dark-theme configuration.
- Allows upload of multiple PDFs, processing them synchronously.
- Transparently provides a "View Source Documents Context" toggle to let users dissect the exact chunk the AI retrieved for trust and verification.

---

## 🛠 Setup & Installation

**1. Clone the project**
```bash
git clone <your_repo_link>
cd <project>
```

**2. Virtual Environment Setup**
Protect your system boundaries by creating an isolated environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt --no-cache-dir
```

**4. Configure Environment Secrets**
Create a `.env` file containing your Google GenAI Token:
```env
GOOGLE_API_KEY="your_api_key_here"
```

**5. Launch the Application**
Ensuring Streamlit binds efficiently to the active Python environment:
```bash
python -m streamlit run app.py
```

*Note: All ingested test APIs, uncompiled caches (`__pycache__`), and sensitive `data/` vector spaces are safely quarantined by `.gitignore`.*
