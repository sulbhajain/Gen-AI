# knowledge_base.py
import os
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class KnowledgeBase:
    def __init__(self, data_path="./knowledge"):
        self.embeddings = OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL", os.getenv("OLLAMA_MODEL", "llama3.1")),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vectorstore = self._load_knowledge_base(data_path)
    
    def _load_knowledge_base(self, data_path):
        """Load and index knowledge base documents"""
        faiss_path = os.getenv("VECTOR_DB_PATH", "./faiss_index")

        if os.path.isdir(faiss_path):
            try:
                return FAISS.load_local(
                    faiss_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception:
                pass

        # Load documents from your data source
        documents = self._load_documents(data_path)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create vector store
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

        vectorstore.save_local(faiss_path)
        
        return vectorstore
    
    def search(self, query: str, k: int = 3) -> str:
        """Search knowledge base and return relevant information"""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            
            if not docs:
                return "No relevant information found in knowledge base."
            
            # Combine relevant chunks
            combined_info = "\n\n".join([doc.page_content for doc in docs])
            
            return f"Based on our knowledge base:\n{combined_info}"
            
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
    
    def _load_documents(self, data_path):
        """Load documents from various sources"""
        documents = []
        if not os.path.isdir(data_path):
            return documents

        for root, _, files in os.walk(data_path):
            for filename in files:
                if not filename.lower().endswith((".md", ".txt")):
                    continue
                file_path = os.path.join(root, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    if content:
                        documents.append(
                            Document(page_content=content, metadata={"source": file_path})
                        )
                except OSError:
                    continue

        return documents
