# Intelligent chunking methods for code documentation RAG

A RAG pipeline that processes text corpus into chunks, generates embeddings, and computes retrieval quality metrics on evaluation datasets with questions and golden excerpts.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/stanislavlamos/rag-code-chunker.git
cd rag-code-chunker
```

2. Install dependencies:
```bash
make install
```

## Running the Application

The application consists of two components:
- FastAPI backend server (runs on http://127.0.0.1:8000)
- Streamlit frontend server (runs on http://127.0.0.1:8501)

### Running Both Servers

To run both the backend and frontend servers:
```bash
make run
```

### Running Servers Separately

To run only the backend server:
```bash
make backend
```

To run only the frontend server:
```bash
make frontend
```

### Other Useful Commands

- Clean up Python cache files:
```bash
make clean
```

- Show available commands:
```bash
make help
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:
```
BACKEND_LOCAL_URL="http://127.0.0.1:8000"
```

## Project Structure

```
.
├── [backend/](backend/)                    # Backend directory
│   ├── [api/](backend/api/)               # FastAPI backend code
│   │   ├── [main.py](backend/api/main.py) # FastAPI application entry point
│   │   └── [query_retriever_multiple.py](backend/api/query_retriever_multiple.py) # Query retriever endpoint
│   ├── [pipeline.py](backend/pipeline.py)  # Main pipeline implementation
│   └── ...
├── [frontend/](frontend/)                  # Frontend directory
│   └── [app.py](frontend/app.py)          # Streamlit frontend application
├── [data/](data/)                         # Data directory
├── [Makefile](Makefile)                   # Build and run commands
├── [requirements.txt](requirements.txt)    # Python dependencies
└── [.env](.env)                          # Environment variables
```

## Features

- Multiple dataset support (chatlogs, state_of_the_union, wikitext)
- Configurable chunking methods
- Multiple embedding model options
- Adjustable chunk size and retrieval parameters
- Real-time pipeline execution
- Interactive results visualization