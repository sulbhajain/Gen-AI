
# app.py - Streamlit web interface
import streamlit as st
from agent import CustomerServiceAgent
import asyncio

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = CustomerServiceAgent()
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Page configuration
st.set_page_config(
    page_title="AI Customer Service Agent",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:
    st.title("🤖 AI Agent")
    st.markdown("---")
    st.markdown("**Features:**")
    st.markdown("• Knowledge base search")
    st.markdown("• Order status checking")
    st.markdown("• Human escalation")
    st.markdown("• Conversation memory")
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.agent.memory.clear()
        st.rerun()

# Main interface
st.title("Customer Service AI Agent")
st.markdown("Ask me anything about your orders, our policies, or general questions!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you today?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.chat(prompt)
        st.markdown(response)
    
    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})

