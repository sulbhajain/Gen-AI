"""Joke generation tool that uses episodic memory as inspiration."""

from typing import Any

from agent.memory import EpisodicMemory


joke_tool = {
    "name": "generate_joke",
    "description": "Generate a light, safe joke about a given topic using episodic memory for inspiration.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The topic to base the joke on"
            }
        },
        "required": ["topic"]
    }
}


async def handle_generate_joke(arguments: dict[str, Any]) -> list[dict]:
    """Generate a joke using episodic memory as inspiration.

    Args:
        arguments: Dict with 'topic' key

    Returns:
        List with text content block containing the joke
    """
    topic = arguments.get("topic")

    if not topic:
        return [{"type": "text", "text": "Error: Missing required argument 'topic'"}]

    topic_text = str(topic).strip()
    if not topic_text:
        return [{"type": "text", "text": "Error: 'topic' cannot be empty"}]

    memory_hint = ""
    try:
        memory = EpisodicMemory()
        episodes = memory.recall_similar(
            task=f"joke about {topic_text}",
            user_id=None,
            k=3,
            min_similarity=0.0
        )
        if episodes:
            hint = episodes[0].get("task") or episodes[0].get("outcome") or ""
            if hint:
                memory_hint = f"Memory hint: {hint}"
    except Exception:
        memory_hint = ""

    setup = f"Why did the {topic_text} bring a notebook?"
    punchline = "To keep track of all the punchlines it kept forgetting."

    joke_lines = [setup, punchline]
    if memory_hint:
        joke_lines.append("")
        joke_lines.append(memory_hint)

    return [{"type": "text", "text": "\n".join(joke_lines)}]
