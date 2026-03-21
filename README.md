# Clinical RAG Assistant using Endee Vector Database

## Project Overview and Problem Statement
The volume of medical and clinical data in modern healthcare presents a significant challenge for researchers, administrators, and continuous learners who need to parse through complex PDF guidelines, diagnostic histories, and clinical textbooks efficiently.

The **Clinical RAG Assistant** provides a solution by acting as a Retrieval-Augmented Generation (RAG) chatbot specialized in answering queries based *only* on the clinical documents users upload. This drastically reduces hallucination by grounding an LLM heavily on authenticated source guidelines, mitigating risks in the medical context while returning highly precise, cited answers.

## System Design and Technical Approach
This project operates on a robust, locally accessible AI pipeline that heavily utilizes vector embeddings for contextual semantic retrieval:
1. **Frontend**: A Flask-based web server hosting an interactive chatbot interface where users can drag-and-drop clinical PDFs.
2. **Ingestion & Chunking**: PDFs are ingested via LangChain's `PyPDFLoader` and logically split using a `RecursiveCharacterTextSplitter`.
3. **Embeddings**: Sentence embeddings are generated locally utilizing the HuggingFace `all-MiniLM-L6-v2` transformer model directly encoded onto the CPU.
4. **Hybrid Retrieval**: We implement a powerful Hybrid Retriever utilizing **Reciprocal Rank Fusion (RRF)**. Semantic (dense) search executes through our Vector Database, while keyword (sparse) text searches handle domain-specific medical acronyms via `BM25Okapi`. 
5. **Generative LLM**: A generative language model accepts the tightly retrieved document chunks alongside the prompt and produces citations directly referring back to the specific chunk's origin PDF and page number algorithmically.

## Explanation of how Endee is used
Endee (`endee-io/endee`) operates as the **core Vector Database** powering the dense semantic search phase of the application pipeline. 

### Why Endee?
Endee manages indexing large arrays of densely-encoded contextual dimensions. Specifically:
- **Index Management**: When the system starts, it utilizes the native `endee.Endee()` Python client to connect to the Endee vector search engine, requesting the retrieval of `(or creation of)` a high-performance Cosine-distance `clinical_rag` index mapping exactly to 384 dimensions.
- **Document Ingestion**: Over the ingestion lifecycle, Endee `.add()` seamlessly inserts the user's chunked PDF texts with payload metadata and generated transformer embeddings into the database under isolated UUIDs.
- **Query Resolution**: At query time, Endee executes an extremely fast top-K nearest-neighbor `.query()` sweep of the dense embeddings generated from the user's question, reliably surfacing the most semantically pertinent text fragments necessary for the LLM context limits.

## Setup and Execution Instructions

### Prerequisites
1. **Clone the Endee Repository Fork** 
   You must clone this project from the forked replica of the core `endee-io/endee` repository.
2. **Start the Endee Database**
   Since Endee is a C++ natively compiled database, follow its built-in launch scripts before booting the Python API:
   ```bash
   chmod +x ./install.sh ./run.sh
   ./install.sh --release --avx2
   ./run.sh
   # Endee listens inherently on port 8080
   ```
3. **Environment Tokens**
   Copy the provided `.env.example` into a local `.env` file and insert your respective Groq or OpenAI API key.

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Ingesting Documents
Place any clinical `.pdf` files you wish to include in your initial corpus into the `/data/` directory, and run:
```bash
python ingest.py
```

### Running the Assistant
Once documents are ingested successfully by the Endee database, initiate the user-facing app:
```bash
python app.py
```
Open `http://localhost:5000` inside your clinical workstation's web-browser to begin securely questioning your sources.
