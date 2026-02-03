# Agent Basic (Ollama + FAISS)

A local customer service agent built with Streamlit, Ollama, and a free FAISS vector database.

## Features
- Knowledge base search (local FAISS index)
- Order status checking (mocked)
- Human escalation (mocked)
- Conversation memory

## Requirements
- macOS with Ollama installed
- Python 3.10+
- uv

## Setup
1. Start Ollama:
   - `ollama serve`
2. Pull a model (once):
   - `ollama pull llama3.1`
3. Install deps:
   - `uv sync`

## Run
- Streamlit UI:
  - `uv run streamlit run app.py`
- CLI agent:
  - `uv run python agent.py`

## Configuration
Edit [.env](.env):
- `OLLAMA_BASE_URL` (default: http://localhost:11434)
- `OLLAMA_MODEL` (default: llama3.1)
- `OLLAMA_EMBEDDING_MODEL` (default: llama3.1)
- `VECTOR_DB_PATH` (default: ./faiss_index)

## Knowledge Base
Add documents to [knowledge](knowledge) as .md or .txt files. The FAISS index is created at first run and saved to `VECTOR_DB_PATH`.

## Notes
- Watchdog is optional but improves Streamlit hot reload performance.
