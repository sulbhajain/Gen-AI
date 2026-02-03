# Runbook: Local Agent Usage

## Start Services
1. Ensure Ollama is running: `ollama serve`
2. Pull a model (once): `ollama pull llama3.1`

## Run UI
- `uv run streamlit run app.py`

## Configuration
- OLLAMA_BASE_URL: http://localhost:11434
- OLLAMA_MODEL: llama3.1
- VECTOR_DB_PATH: ./faiss_index

## Troubleshooting
- If Streamlit is missing, run with `uv run`.
- If Ollama port is busy, it’s already running.
