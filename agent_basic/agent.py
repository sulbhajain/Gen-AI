# agent.py
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain.tools import tool
import os
from knowledge_base import KnowledgeBase


class SimpleMemory:
    def __init__(self, k: int = 10):
        self.k = k
        self.messages: list[dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        max_len = self.k * 2
        if len(self.messages) > max_len:
            self.messages = self.messages[-max_len:]

    def clear(self) -> None:
        self.messages = []

class CustomerServiceAgent:
    def __init__(self):
        self.llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3.1"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=float(os.getenv("TEMPERATURE", "0.1"))
        )
        
        self.memory = SimpleMemory(k=10)
        self.knowledge_base = KnowledgeBase()
        
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self):
        """Define the tools available to the agent"""
        @tool("search_knowledge_base", description="Search the company knowledge base for information")
        def search_knowledge_base_tool(query: str) -> str:
            return self.search_knowledge_base(query)

        @tool("escalate_to_human", description="Escalate complex issues to human agents")
        def escalate_to_human_tool(reason: str) -> str:
            return self.escalate_to_human(reason)

        @tool("check_order_status", description="Check the status of a customer order by order ID")
        def check_order_status_tool(order_id: str) -> str:
            return self.check_order_status(order_id)

        return [
            search_knowledge_base_tool,
            escalate_to_human_tool,
            check_order_status_tool,
        ]
    
    def _create_agent(self):
        """Create the agent with prompt and tools"""
        return create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self._get_system_prompt(),
            debug=True,
        )
    
    def _get_system_prompt(self):
        return """You are a helpful customer service agent for TechCorp. 
        Your role is to assist customers with their inquiries professionally and efficiently.
        
        Guidelines:
        - Always be polite and empathetic
        - Use available tools to find accurate information
        - If you cannot resolve an issue, escalate to a human agent
        - Keep responses concise but informative
        - Ask clarifying questions when needed
        
        Available tools:
        1. search_knowledge_base: Find information in the company knowledge base
        2. escalate_to_human: Transfer complex issues to human agents
        3. check_order_status: Look up order information by order ID
        """
    
    def search_knowledge_base(self, query: str) -> str:
        """Search the knowledge base for relevant information"""
        try:
            return self.knowledge_base.search(query)
        except Exception:
            return "I couldn't find specific information about that. Let me escalate this to a human agent."
    
    def escalate_to_human(self, reason: str) -> str:
        """Escalate the conversation to a human agent"""
        return f"I'm transferring you to a human agent who can better assist with: {reason}. Please hold while I connect you."
    
    def check_order_status(self, order_id: str) -> str:
        """Check order status by order ID"""
        # Mock order status check
        if order_id.startswith("ORD"):
            return f"Order {order_id} is currently being processed and will ship within 2 business days."
        else:
            return "Please provide a valid order ID starting with 'ORD'."
    
    def chat(self, message: str) -> str:
        """Process a user message and return agent response"""
        try:
            self.memory.add("user", message)
            result = self.agent.invoke({"messages": self.memory.messages})
            messages = result.get("messages", [])
            assistant_text = ""

            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    assistant_text = msg.get("content", "")
                    break
                if hasattr(msg, "type") and getattr(msg, "type") == "ai":
                    assistant_text = getattr(msg, "content", "")
                    break

            if not assistant_text:
                assistant_text = "I'm sorry, I couldn't generate a response."

            self.memory.add("assistant", assistant_text)
            return assistant_text
        except Exception as e:
            return f"I'm sorry, I encountered an error: {str(e)}. Let me escalate this to a human agent."

# Usage example
if __name__ == "__main__":
    agent = CustomerServiceAgent()
    
    # Test the agent
    print("Customer Service Agent initialized!")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("Customer: ")
        if user_input.lower() == 'quit':
            break
        
        response = agent.chat(user_input)
        print(f"Agent: {response}\n")
