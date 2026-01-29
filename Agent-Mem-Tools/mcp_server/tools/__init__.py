"""Tool implementations for the MCP server."""

from .calculator import calculator_tools, handle_calculator
from .file_ops import file_tools, handle_read_file, handle_write_file, handle_list_files
from .weather import weather_tool, handle_get_weather
from .jokes import joke_tool, handle_generate_joke
from .codegen import codegen_tool, handle_generate_python_code

__all__ = [
    "calculator_tools",
    "handle_calculator",
    "file_tools",
    "handle_read_file",
    "handle_write_file",
    "handle_list_files",
    "weather_tool",
    "handle_get_weather",
    "joke_tool",
    "handle_generate_joke",
    "codegen_tool",
    "handle_generate_python_code",
]
