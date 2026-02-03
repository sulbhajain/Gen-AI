# test_setup.py
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

# Load environment variables
load_dotenv()

# Test Ollama connection
try:
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3.1"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("TEMPERATURE", "0.1"))
    )
    response = llm.invoke("Hello, world!")
    print("✅ Ollama connection successful")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"❌ Ollama error: {e}")

print("🎉 Setup verification complete!")
