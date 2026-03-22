# Clinical RAG Assistant using Endee

A chatbot that answers clinical questions from uploaded PDFs using the **Endee** vector database and **Retrieval-Augmented Generation (RAG)**.

## My Contribution

This project was implemented by me as part of the **Endee AI/ML evaluation process**.

I extended the base Endee repository and built a complete AI application including:

- Clinical document ingestion pipeline (PDF → text chunks)
- Embedding generation using transformer models
- Integration with Endee vector database for semantic search
- Hybrid retrieval (Dense + BM25 using Reciprocal Rank Fusion)
- Flask-based chatbot interface
- Citation-based answer generation from source documents

## Project Overview and Problem Statement

The volume of medical and clinical data in modern healthcare presents a significant challenge for researchers, administrators, and learners who need to quickly extract relevant insights from large documents.

This project addresses that challenge by building a **Clinical RAG Assistant** that answers queries based **only on trusted uploaded documents**, thereby reducing hallucinations and improving reliability.

## Solution

The Clinical RAG Assistant:

- Accepts clinical PDF documents
- Converts them into embeddings
- Stores them in the Endee vector database
- Retrieves relevant document chunks
- Generates accurate, citation-backed answers

## System Design and Technical Approach

### Frontend
- Flask-based web interface
- Allows users to upload PDFs and ask questions

### Ingestion & Chunking
- PDFs processed using **LangChain** `PyPDFLoader`
- Split into chunks using `RecursiveCharacterTextSplitter`

### Embeddings
- Generated using Hugging Face model: **`all-MiniLM-L6-v2`**
- Runs locally on CPU

### Hybrid Retrieval
- Dense search using **Endee** vector database
- Sparse (keyword) search using **BM25**
- Combined using **Reciprocal Rank Fusion (RRF)**

### LLM Generation
- Uses **Groq** or **OpenAI** API
- Generates answers grounded in retrieved context
- Provides source-based citations

## How Endee is Used

Endee serves as the **core vector database** in this system.

**Key Roles:**

- **Index Management**
  - Creates and manages a vector index of **384 dimensions**
- **Document Storage**
  - Stores embeddings of document chunks along with metadata
- **Query Processing**
  - Performs fast top-K similarity search using `.query()`
- **Semantic Retrieval**
  - Returns the most relevant chunks to feed into the LLM

## Tech Stack

- Python
- Flask
- LangChain
- HuggingFace Transformers
- Endee Vector Database
- BM25 (`rank-bm25`)
- Groq / OpenAI API

## Setup and Execution Instructions

### 1. Clone Repository

```bash
git clone https://github.com/jyothir-369/endee_clinical_rag_assistant.git
cd endee_clinical_rag_assistant
2. Start Endee Database
Bashchmod +x ./install.sh ./run.sh
./install.sh --release --avx2
./run.sh
Endee will be available at:
http://localhost:8080
3. Setup Environment
Bashcp .env.example .env
Edit .env and add your API key:
envOPENAI_API_KEY=your_key_here
(Use Groq API key if you're using Groq instead)
4. Install Dependencies
Bashpip install -r requirements.txt
5. Ingest Documents
Place your clinical PDFs in the /data/ folder, then run:
Bashpython ingest.py
6. Run the Application
Bashpython app.py
Open in your browser:
http://localhost:5000
Features

Semantic search using vector embeddings
Hybrid retrieval (dense + keyword)
Clinical document understanding
Citation-based, grounded answers
Simple and clean web UI

Future Improvements

Support for more file formats (DOCX, PPTX, etc.)
Improved UI/UX
User authentication
Cloud-based deployment

Author
Jyothir
