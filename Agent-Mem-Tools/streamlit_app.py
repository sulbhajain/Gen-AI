#!/usr/bin/env python3
"""Streamlit UI for the Agentic AI with Memory Demo."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from agent import create_agent_graph
from agent.state import create_initial_state
from agent.memory import LongTermMemory, EpisodicMemory, reset_checkpointer


load_dotenv()


def _require_api_key() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file or environment."  # noqa: E501
        )
        st.stop()


def _init_session_state() -> None:
    if "graph" not in st.session_state:
        st.session_state.graph = create_agent_graph()
    if "user_id" not in st.session_state:
        st.session_state.user_id = "demo_user"
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"{st.session_state.user_id}_{int(time.time())}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0


def _extract_ai_text(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            if isinstance(msg.content, str):
                return msg.content
            if isinstance(msg.content, list):
                parts: list[str] = []
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                return "\n".join(parts)
    return "I completed the task but have nothing to add."


async def _run_agent_async(
    user_input: str,
    store_episode: bool,
) -> str:
    initial_state = create_initial_state(
        user_id=st.session_state.user_id,
        task=user_input,
        user_message=user_input,
    )
    initial_state["should_store_episode"] = store_episode
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    result = await st.session_state.graph.ainvoke(initial_state, config)
    return _extract_ai_text(result.get("messages", []))


def _run_agent_sync(user_input: str, store_episode: bool) -> str:
    return asyncio.run(_run_agent_async(user_input, store_episode))


def _show_memory_summary() -> dict[str, Any]:
    long_term = LongTermMemory()
    episodic = EpisodicMemory()

    context = long_term.get_user_context(st.session_state.user_id)
    stats = episodic.get_stats(st.session_state.user_id)
    episodes = episodic.get_user_episodes(st.session_state.user_id, limit=5)

    return {
        "preferences": context.get("preferences", {}),
        "facts": context.get("facts", []),
        "episodic_stats": stats,
        "recent_episodes": episodes,
    }


def _start_new_thread() -> None:
    st.session_state.thread_id = f"{st.session_state.user_id}_{int(time.time())}"
    st.session_state.interaction_count = 0
    st.session_state.messages = []


def _clear_memory() -> None:
    long_term = LongTermMemory()
    episodic = EpisodicMemory()
    long_term.clear_user_data(st.session_state.user_id)
    episodic.clear_user_episodes(st.session_state.user_id)
    reset_checkpointer()
    _start_new_thread()


st.set_page_config(
    page_title="Agentic AI with Memory",
    page_icon="🧠",
    layout="wide",
)

st.title("Agentic AI with Memory")

_require_api_key()
_init_session_state()

with st.sidebar:
    st.header("Session")
    st.text(f"User ID: {st.session_state.user_id}")
    st.text(f"Thread ID: {st.session_state.thread_id}")

    if st.button("New thread", use_container_width=True):
        _start_new_thread()
        st.success("Started a new thread.")

    if st.button("Show memory", use_container_width=True):
        memory = _show_memory_summary()
        st.subheader("Long-term memory")
        st.write("Preferences")
        st.json(memory.get("preferences", {}))
        st.write("Facts")
        st.json(memory.get("facts", []))

        st.subheader("Episodic memory")
        st.json(memory.get("episodic_stats", {}))
        st.write("Recent episodes")
        st.json(memory.get("recent_episodes", []))

    if st.button("Clear memory", use_container_width=True):
        _clear_memory()
        st.warning("Cleared memories and started a new thread.")

st.markdown(
    """
This Streamlit app provides a chat interface for the agentic workflow using
LangGraph, MCP tools, and multi-layer memory.
"""
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask the agent anything...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.interaction_count += 1
    store_episode = st.session_state.interaction_count % 3 == 0

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = _run_agent_sync(prompt, store_episode)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
