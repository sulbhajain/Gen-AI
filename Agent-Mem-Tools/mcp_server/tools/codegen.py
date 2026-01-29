"""Python code generation tool."""

from typing import Any


codegen_tool = {
    "name": "generate_python_code",
    "description": "Generate a Python code snippet based on a prompt.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Description of the Python code to generate"
            }
        },
        "required": ["prompt"]
    }
}


async def handle_generate_python_code(arguments: dict[str, Any]) -> list[dict]:
    """Generate a Python code snippet from a prompt."""
    prompt = arguments.get("prompt")

    if not prompt:
        return [{"type": "text", "text": "Error: Missing required argument 'prompt'"}]

    prompt_text = str(prompt).strip()
    if not prompt_text:
        return [{"type": "text", "text": "Error: 'prompt' cannot be empty"}]

    code = (
        "# Generated Python snippet\n"
        f"# Prompt: {prompt_text}\n\n"
        "def main():\n"
        "    print(\"Hello from generated code!\")\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    return [{"type": "text", "text": code}]
