# PDF RAG Service

A simple Retrieval-Augmented Generation (RAG) API built with FastAPI, ChromaDB, Sentence Transformers, and the Groq LLM API.

The service accepts a PDF upload, converts it into text chunks, stores embeddings in a local ChromaDB vector store, and answers questions by retrieving relevant PDF context and querying an LLM.

## Features

- Upload a PDF and index it automatically
- Store text embeddings in `chroma_store`
- Ask questions against the uploaded PDF using an LLM
- FastAPI endpoints for uploading and querying

## Prerequisites

- Python 3.10+ recommended
- A Groq API key for the LLM
- Internet access for model download and Groq API usage

## Setup

1. Clone or open the repository in your workspace.
2. Install dependencies:

```bash
python -m pip install fastapi uvicorn python-dotenv sentence-transformers chromadb groq langchain-community langchain-text-splitters pypdf
```

3. Create a `.env` file in the project root with your Groq API key:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

4. Make sure the following directories are present (they are created automatically if missing):

- `uploads/`
- `chroma_store/`

## Running the service

Start the FastAPI app with Uvicorn:

```bash
uvicorn app:app --reload
```

By default, the API runs on `http://127.0.0.1:8000`.

## API Endpoints

### Upload PDF

- URL: `POST /upload-pdf/`
- Body: `multipart/form-data` with a file field named `file`
- Description: Uploads a PDF, clears previous uploads, processes the PDF, and stores embeddings.

Example curl:

```bash
curl -X POST "http://127.0.0.1:8000/upload-pdf/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your.pdf"
```

Response:

```json
{
  "message": "PDF uploaded and indexed successfully"
}
```

### Ask a question

- URL: `GET /ask/`
- Query parameter: `query`
- Description: Retrieves the top-matching PDF text chunks and asks the LLM to answer based only on that context.

Example curl:

```bash
curl "http://127.0.0.1:8000/ask/?query=What+is+the+main+topic+of+the+document%3F"
```

Response:

```json
{
  "query": "What is the main topic of the document?",
  "answer": "..."
}
```

## Project structure

- `app.py` - FastAPI application exposing upload and question endpoints
- `rag.py` - PDF processing, chunking, embedding, retrieval, and LLM prompt logic
- `chroma_store/` - local ChromaDB persistent store directory
- `uploads/` - temporary upload storage for the latest PDF

## Notes

- The uploaded PDF is saved to `uploads/` and the collection is rebuilt each time a new PDF is uploaded.
- `rag.py` uses `all-MiniLM-L6-v2` for embedding generation.
- The `ask_question` function currently uses `llama-3.1-8b-instant` through Groq.
- If the answer is not found in the retrieved document fragments, the LLM may still attempt to answer. Use the API response carefully.

## Troubleshooting

- If the app cannot find your Groq API key, verify `.env` contains `GROQ_API_KEY` and that `python-dotenv` loads it correctly.
- If model downloads fail, ensure internet connectivity and that `sentence-transformers` dependencies are installed.
- If the vector store becomes inconsistent, delete `chroma_store/` and re-run the upload.
