"""Graph nodes for the LangGraph agent workflow.

Each node is a function that takes the current state and returns
updates to be merged into the state.
"""

import json
import re
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, ToolMessage, SystemMessage, HumanMessage

from .state import AgentState
from .memory import LongTermMemory, EpisodicMemory


# Initialize memory systems (singletons)
_long_term_memory: LongTermMemory | None = None
_episodic_memory: EpisodicMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    """Get or create the long-term memory instance."""
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
    return _long_term_memory


def get_episodic_memory() -> EpisodicMemory:
    """Get or create the episodic memory instance."""
    global _episodic_memory
    if _episodic_memory is None:
        _episodic_memory = EpisodicMemory()
    return _episodic_memory


# Initialize the Claude model
def get_model() -> ChatAnthropic:
    """Get the Claude model instance."""
    return ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0.7,
        max_tokens=4096,
    )


# Define tools for the agent
TOOLS = [
    {
        "name": "calculator",
        "description": "Perform basic arithmetic operations (add, subtract, multiply, divide) on two numbers",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "The arithmetic operation to perform"
                },
                "a": {"type": "number", "description": "The first operand"},
                "b": {"type": "number", "description": "The second operand"}
            },
            "required": ["operation", "a", "b"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file from the workspace directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to read (relative to workspace)"
                }
            },
            "required": ["filename"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the workspace directory",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "The name of the file to write (relative to workspace)"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file"
                }
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "list_files",
        "description": "List all files in the workspace directory",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_weather",
        "description": "Get the current weather for a city. Returns temperature (Fahrenheit and Celsius), condition, and humidity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to get weather for"
                }
            },
            "required": ["city"]
        }
    },
    {
        "name": "generate_joke",
        "description": "Generate a light, safe joke about a given topic using episodic memory for inspiration.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to base the joke on"
                }
            },
            "required": ["topic"]
        }
    },
    {
        "name": "generate_python_code",
        "description": "Generate a Python code snippet based on a prompt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Description of the Python code to generate"
                }
            },
            "required": ["prompt"]
        }
    },
    {
        "name": "store_user_preference",
        "description": "Store a user preference for future conversations (e.g., temperature unit, language)",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The preference key (e.g., 'temperature_unit', 'name')"
                },
                "value": {
                    "type": "string",
                    "description": "The preference value (e.g., 'celsius', 'Alex')"
                }
            },
            "required": ["key", "value"]
        }
    },
    {
        "name": "store_user_fact",
        "description": "Store a fact about the user for future reference (e.g., workplace, interests)",
        "input_schema": {
            "type": "object",
            "properties": {
                "fact_type": {
                    "type": "string",
                    "description": "Category of fact (e.g., 'personal', 'work', 'interest')"
                },
                "content": {
                    "type": "string",
                    "description": "The fact to store (e.g., 'Works at Acme Corp')"
                }
            },
            "required": ["fact_type", "content"]
        }
    }
]


def build_system_prompt(state: AgentState) -> str:
    """Build the system prompt including memory context.

    Args:
        state: Current agent state with memory loaded

    Returns:
        System prompt string
    """
    prompt_parts = [
        "You are a helpful AI assistant with access to tools and memory capabilities.",
        "",
        "## Your Capabilities",
        "- Calculator for math operations",
        "- File operations (read, write, list) in the workspace",
        "- Weather lookup for cities",
        "- Joke generation for a given topic",
        "- Python code generation from a prompt",
        "- Store user preferences and facts for future conversations",
        "",
        "## Guidelines",
        "- Use tools when needed to complete tasks",
        "- Store important user information using store_user_preference or store_user_fact only when explicitly stated by the user",
        "- Reference past experiences when relevant",
        "- Be concise but helpful",
    ]

    # Add user preferences if available
    if state.get("user_preferences"):
        prompt_parts.append("")
        prompt_parts.append("## User Preferences (from long-term memory)")
        for key, value in state["user_preferences"].items():
            prompt_parts.append(f"- {key}: {value}")

    # Add known facts if available
    if state.get("known_facts"):
        prompt_parts.append("")
        prompt_parts.append("## Known Facts About User (from long-term memory)")
        for fact in state["known_facts"][:10]:  # Limit to 10 facts
            prompt_parts.append(f"- {fact}")

    # Add similar past experiences if available
    if state.get("similar_episodes"):
        prompt_parts.append("")
        prompt_parts.append("## Similar Past Experiences (from episodic memory)")
        prompt_parts.append("Use these as guidance for how to approach similar tasks:")
        for episode in state["similar_episodes"][:3]:  # Limit to 3 episodes
            prompt_parts.append(f"- Task: {episode.get('task', 'Unknown')}")
            actions = episode.get('actions', [])
            if actions:
                prompt_parts.append(f"  Actions taken: {', '.join(actions)}")
            prompt_parts.append(f"  Outcome: {episode.get('outcome', 'Unknown')}")

    return "\n".join(prompt_parts)


# ===== Node Functions =====

async def load_user_context(state: AgentState) -> dict[str, Any]:
    """Node: Load user context from long-term memory.

    This runs at the start of each conversation to load
    the user's preferences and known facts.
    """
    print("[Memory] Loading user context from long-term memory...")

    memory = get_long_term_memory()
    user_context = memory.get_user_context(state["user_id"])

    preferences = user_context.get("preferences", {})
    facts = user_context.get("facts", [])

    print(f"[Memory] Loaded {len(preferences)} preferences, {len(facts)} facts")

    return {
        "user_preferences": preferences,
        "known_facts": facts
    }


async def retrieve_episodes(state: AgentState) -> dict[str, Any]:
    """Node: Retrieve similar past experiences from episodic memory.

    This finds past tasks similar to the current one to provide
    few-shot guidance to the agent.
    """
    print("[Memory] Searching for similar past experiences...")

    memory = get_episodic_memory()
    current_task = state.get("current_task", "")

    if not current_task:
        print("[Memory] No current task set, skipping episode retrieval")
        return {"similar_episodes": []}

    episodes = memory.recall_similar(
        task=current_task,
        user_id=state["user_id"],
        k=3,
        keyword_query=current_task
    )

    print(f"[Memory] Found {len(episodes)} similar episodes")
    for ep in episodes:
        print(f"  - {ep.get('task', 'Unknown')[:50]}... (similarity: {ep.get('similarity', 0):.2f})")

    return {"similar_episodes": episodes}


async def call_agent(state: AgentState) -> dict[str, Any]:
    """Node: Call Claude with the current context and tools.

    This is the main reasoning node where Claude decides
    what to do next.
    """
    print("[Agent] Calling Claude for reasoning...")

    model = get_model()
    model_with_tools = model.bind_tools(TOOLS)

    # Build system prompt with memory context
    system_prompt = build_system_prompt(state)

    # Prepare messages with system prompt
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])

    # Call the model
    response = await model_with_tools.ainvoke(messages)

    print(f"[Agent] Response type: {type(response).__name__}")
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[Agent] Tool calls: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": [response]}


async def execute_tools(state: AgentState) -> dict[str, Any]:
    """Node: Execute tool calls from the agent's response.

    This handles all tool execution and returns results
    back to the agent.
    """
    print("[Tools] Executing tool calls...")

    # Import tool handlers
    from mcp_server.tools import (
        handle_calculator,
        handle_read_file,
        handle_write_file,
        handle_list_files,
        handle_get_weather,
        handle_generate_joke,
        handle_generate_python_code,
    )

    # Get the last message (should be AIMessage with tool_calls)
    last_message = state["messages"][-1]

    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        print("[Tools] No tool calls to execute")
        return {"messages": []}

    tool_messages = []
    actions_taken = list(state.get("task_actions", []))
    task_metadata = dict(state.get("task_metadata", {}))
    should_store_episode = bool(state.get("should_store_episode", False))
    task_outcome = state.get("task_outcome", "")
    memory = get_long_term_memory()

    def _last_user_message_text() -> str:
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                return str(msg.content or "")
        return ""

    def _value_tokens_in_message(message: str, value: str) -> bool:
        msg_tokens = re.findall(r"[a-z0-9]+", message.lower())
        val_tokens = [t for t in re.findall(r"[a-z0-9]+", str(value).lower()) if t]
        if not val_tokens:
            return False
        return all(t in msg_tokens for t in val_tokens)

    def _explicit_preference_in_message(message: str, value: str) -> bool:
        msg = message.lower()
        if not _value_tokens_in_message(message, value):
            return False
        triggers = [
            "i prefer",
            "my preference",
            "i like",
            "i love",
            "my favorite",
            "i'd like",
            "i want",
            "my name is",
            "name is",
            "i am",
            "i'm",
        ]
        return any(t in msg for t in triggers)

    last_user_message = _last_user_message_text()

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        print(f"[Tools] Executing: {tool_name}({json.dumps(tool_args)})")

        # Route to appropriate handler
        try:
            if tool_name == "calculator":
                result = await handle_calculator(tool_args)
                actions_taken.append(f"calculator({tool_args.get('operation')})")

            elif tool_name == "read_file":
                result = await handle_read_file(tool_args)
                actions_taken.append(f"read_file({tool_args.get('filename')})")

            elif tool_name == "write_file":
                result = await handle_write_file(tool_args)
                actions_taken.append(f"write_file({tool_args.get('filename')})")

            elif tool_name == "list_files":
                result = await handle_list_files(tool_args)
                actions_taken.append("list_files()")

            elif tool_name == "get_weather":
                result = await handle_get_weather(tool_args)
                actions_taken.append(f"get_weather({tool_args.get('city')})")
                
            elif tool_name == "generate_joke":
                result = await handle_generate_joke(tool_args)
                topic = tool_args.get("topic")
                actions_taken.append(f"generate_joke({topic})")
                if topic:
                    task_metadata["joke_topic"] = str(topic)
                if not task_outcome:
                    task_outcome = result[0]["text"] if result else "Joke generated"
                # Store joke episode immediately to avoid routing misses
                if not task_metadata.get("episode_stored"):
                    task = state.get("current_task", "")
                    if task and actions_taken:
                        episodic = get_episodic_memory()
                        episodic.store_episode(
                            user_id=state["user_id"],
                            task=task,
                            actions=actions_taken,
                            outcome=task_outcome or "Joke generated",
                            success=True,
                            metadata=task_metadata,
                        )
                        task_metadata["episode_stored"] = True
                # Ensure store node doesn't duplicate this episode
                should_store_episode = False

            elif tool_name == "generate_python_code":
                result = await handle_generate_python_code(tool_args)
                actions_taken.append(f"generate_python_code({tool_args.get('prompt')})")

            elif tool_name == "store_user_preference":
                # Handle preference storage
                key = tool_args.get("key", "")
                value = tool_args.get("value", "")
                if _explicit_preference_in_message(last_user_message, str(value)):
                    memory.store_preference(state["user_id"], key, value)
                    result = [{"type": "text", "text": f"Stored preference: {key} = {value}"}]
                    actions_taken.append(f"store_preference({key})")
                else:
                    result = [{
                        "type": "text",
                        "text": "Skipped storing preference: not explicitly stated by the user."
                    }]

            elif tool_name == "store_user_fact":
                # Handle fact storage
                fact_type = tool_args.get("fact_type", "general")
                content = tool_args.get("content", "")
                if _value_tokens_in_message(last_user_message, str(content)):
                    memory.store_fact(state["user_id"], fact_type, content, source="user_stated")
                    result = [{"type": "text", "text": f"Stored fact: {content}"}]
                    actions_taken.append(f"store_fact({fact_type})")
                else:
                    result = [{
                        "type": "text",
                        "text": "Skipped storing fact: not explicitly stated by the user."
                    }]

            else:
                result = [{"type": "text", "text": f"Unknown tool: {tool_name}"}]

            # Extract text from result
            result_text = result[0]["text"] if result else "No result"

        except Exception as e:
            result_text = f"Error executing {tool_name}: {str(e)}"
            print(f"[Tools] Error: {result_text}")

        print(f"[Tools] Result: {result_text[:100]}...")

        # Create tool message
        tool_messages.append(
            ToolMessage(content=result_text, tool_call_id=tool_id)
        )

    return {
        "messages": tool_messages,
        "task_actions": actions_taken,
        "task_metadata": task_metadata,
        "task_outcome": task_outcome,
        "should_store_episode": should_store_episode
    }


async def store_episode(state: AgentState) -> dict[str, Any]:
    """Node: Store the completed task as an episode in episodic memory.

    This runs after a task is completed to save the experience
    for future reference.
    """
    # Only store if flagged
    if not state.get("should_store_episode", False):
        print("[Memory] Skipping episode storage (not flagged)")
        return {}

    print("[Memory] Storing episode to episodic memory...")

    memory = get_episodic_memory()

    task = state.get("current_task", "")
    actions = state.get("task_actions", [])
    outcome = state.get("task_outcome", "Task completed")
    metadata = state.get("task_metadata", {})

    if metadata.get("episode_stored"):
        print("[Memory] Episode already stored earlier in this run")
        return {"should_store_episode": False}

    if task and actions:
        episode_id = memory.store_episode(
            user_id=state["user_id"],
            task=task,
            actions=actions,
            outcome=outcome,
            success=True
            ,metadata=metadata
        )
        print(f"[Memory] Stored episode: {episode_id}")
    else:
        print("[Memory] No task/actions to store")

    return {"should_store_episode": False}


def route_agent(state: AgentState) -> Literal["tools", "store", "end"]:
    """Routing function: Determine next step after agent reasoning.

    Returns:
        - "tools": If the agent wants to use tools
        - "store": If the task is complete and should be stored
        - "end": If done (no tools, no storage needed)
    """
    last_message = state["messages"][-1]

    # Check if agent wants to use tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # If tool calls are only preference/fact storage but not explicitly stated, skip tools
        def _last_user_message_text() -> str:
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    return str(msg.content or "")
            return ""

        def _value_tokens_in_message(message: str, value: str) -> bool:
            msg_tokens = re.findall(r"[a-z0-9]+", message.lower())
            val_tokens = [t for t in re.findall(r"[a-z0-9]+", str(value).lower()) if t]
            if not val_tokens:
                return False
            return all(t in msg_tokens for t in val_tokens)

        def _explicit_preference_in_message(message: str, value: str) -> bool:
            msg = message.lower()
            if not _value_tokens_in_message(message, value):
                return False
            triggers = [
                "i prefer",
                "my preference",
                "i like",
                "i love",
                "my favorite",
                "i'd like",
                "i want",
                "my name is",
                "name is",
                "i am",
                "i'm",
            ]
            return any(t in msg for t in triggers)

        last_user_message = _last_user_message_text()
        invalid_only = True
        for tool_call in last_message.tool_calls:
            name = tool_call.get("name")
            if name == "store_user_preference":
                value = tool_call.get("args", {}).get("value", "")
                if _explicit_preference_in_message(last_user_message, value):
                    invalid_only = False
            elif name == "store_user_fact":
                content = tool_call.get("args", {}).get("content", "")
                if _value_tokens_in_message(last_user_message, content):
                    invalid_only = False
            else:
                invalid_only = False

        if invalid_only:
            print("[Router] -> end")
            return "end"

        print("[Router] -> tools")
        return "tools"

    # Check if we should store this as an episode
    # (has actions and task outcome)
    if state.get("task_actions") and state.get("should_store_episode"):
        print("[Router] -> store")
        return "store"

    print("[Router] -> end")
    return "end"
