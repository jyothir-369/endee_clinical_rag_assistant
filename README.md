# Clinical RAG Assistant using Endee

A chatbot that answers clinical questions from uploaded PDFs using Endee vector database and Retrieval-Augmented Generation (RAG).

---

## My Contribution

This project is implemented by me as part of the Endee AI/ML evaluation process.

I extended the base Endee repository and built a complete AI application including:

- Clinical document ingestion pipeline (PDF → text chunks)
- Embedding generation using transformer models
- Integration with Endee vector database for semantic search
- Hybrid retrieval (Dense + BM25 using Reciprocal Rank Fusion)
- Flask-based chatbot interface
- Citation-based answer generation from source documents

---

## Project Overview and Problem Statement

The volume of medical and clinical data in modern healthcare presents a significant challenge for researchers, administrators, and learners who need to extract relevant insights from large documents quickly.

This project solves that problem by building a Clinical RAG Assistant that answers queries based only on trusted uploaded documents, reducing hallucination and improving reliability.

---

## Solution

The Clinical RAG Assistant:

- Accepts clinical PDF documents
- Converts them into embeddings
- Stores them in Endee vector database
- Retrieves relevant document chunks
- Generates accurate answers with citations

---

## System Design and Technical Approach

This project uses a complete AI pipeline:

### Frontend
- Flask-based web interface
- Allows users to upload PDFs and ask questions

### Ingestion & Chunking
- PDFs processed using LangChain PyPDFLoader
- Split into chunks using RecursiveCharacterTextSplitter

### Embeddings
- Generated using HuggingFace model: all-MiniLM-L6-v2
- Runs locally on CPU

### Hybrid Retrieval
- Dense search using Endee vector database
- Sparse search using BM25
- Combined using Reciprocal Rank Fusion (RRF)

### LLM Generation
- Uses Groq/OpenAI API
- Generates answers using retrieved context
- Provides source-based citations

---

## How Endee is Used

Endee acts as the core vector database in this system.

### Key Roles:

- **Index Management**
  - Creates and manages a vector index of 384 dimensions

- **Document Storage**
  - Stores embeddings of document chunks with metadata

- **Query Processing**
  - Performs fast top-K similarity search using `.query()`

- **Semantic Retrieval**
  - Returns most relevant chunks for LLM input

---

## Tech Stack

- Python
- Flask
- LangChain
- HuggingFace Transformers
- Endee Vector Database
- BM25 (rank-bm25)
- Groq / OpenAI API

---

## Setup and Execution Instructions

### Prerequisites

Clone this repository:

```bash
git clone https://github.com/jyothir-369/endee_clinical_rag_assistant.git
cd endee_clinical_rag_assistant

Start Endee Database
chmod +x ./install.sh ./run.sh
./install.sh --release --avx2
./run.sh

Endee runs on:

http://localhost:8080
Setup Environment

Create .env file:

cp .env.example .env

Add your API key:

OPENAI_API_KEY=your_key_here
Install Dependencies
pip install -r requirements.txt
Ingest Documents

Place your PDFs inside /data/ folder and run:

python ingest.py
Run the Application
python app.py

Open in browser:

http://localhost:5000
Features
Semantic search using vector embeddings
Hybrid retrieval (dense + keyword)
Clinical document understanding
Citation-based answers
Simple web UI for interaction
Future Improvements
Add support for more file formats
Improve UI/UX
Add user authentication
Deploy as cloud-based service
Author

Jyothir



If you want next:
👉 I can review your repo once before submission  
👉 Or help you crack next round 👍
