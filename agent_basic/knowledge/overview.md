# TechCorp Customer Service Agent (Repo Overview)

This project is a Streamlit-based customer service agent powered by Ollama local models.

## Key Components
- app.py: Streamlit UI.
- agent.py: Agent logic and tools.
- knowledge_base.py: Local FAISS vector store for knowledge retrieval.
- .env: Configuration (OLLAMA_BASE_URL, OLLAMA_MODEL, VECTOR_DB_PATH).

## Typical Questions
- How to check an order status (use the check_order_status tool).
- How to escalate to a human agent.
- Where to update model settings.
