"""Enterprise Text-to-SQL with MCP and PostgreSQL.

Solves Problem 1 (Erroneous Data Analysis) and Problem 2 (Schema Linking & Pruning).
Includes native LangChain tool adapters and LCEL prompt templates for agent orchestration.
"""

from __future__ import annotations

from src.langchain_prompts import (
    SQLPlan,
    SYSTEM_PROMPT_TEXT2SQL,
    format_langchain_context,
    get_text2sql_prompt_template,
)
from src.langchain_tools import (
    get_langchain_tool_map,
    get_langchain_tools,
)

__version__ = "0.2.0"

__all__ = [
    "get_langchain_tools",
    "get_langchain_tool_map",
    "get_text2sql_prompt_template",
    "SQLPlan",
    "SYSTEM_PROMPT_TEXT2SQL",
    "format_langchain_context",
]

