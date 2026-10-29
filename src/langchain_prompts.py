"""LangChain Prompt Templates and Structured Output Models for Text-to-SQL.

Defines:
1. SQLPlan: Pydantic model for LangChain .with_structured_output().
2. Text-to-SQL System Prompt guarding against Problem 1 (Erroneous Analysis)
   and Problem 2 (Schema Bloat / Hallucinated Joins).
3. ChatPromptTemplate factory for LCEL chains.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from pydantic import BaseModel, Field


class SQLPlan(BaseModel):
    """Structured analytical plan and verified SQL query produced by LangChain LLM chains."""

    metric_name: str = Field(
        description="Clear business metric name (e.g. 'Staff Attendance Rate', 'Individual Loan Rejections')."
    )
    target_entity: str = Field(
        description="Grain entity identifier (e.g. 'staff_no', 'user_id', 'app_no', 'cluster_name')."
    )
    is_percentage: bool = Field(
        description="True if the metric is a percentage rate, False if count or aggregate currency."
    )
    candidate_tables: List[str] = Field(
        default_factory=list,
        description="List of database tables required for the query after schema pruning.",
    )
    join_predicates: List[str] = Field(
        default_factory=list,
        description="Active join conditions connecting candidate tables discovered via relational graph.",
    )
    sql_query: str = Field(
        description="Semantically sound, read-only SQL query guarded against grain and filter traps."
    )
    explanation: str = Field(
        description="Concise explanation of denominator integrity, distinct entity counting, and safe math."
    )


SYSTEM_PROMPT_TEXT2SQL = """You are an expert Enterprise Text-to-SQL Specialist for PostgreSQL.
Your mission is to translate natural language business questions into semantically sound, executable SQL queries.

You MUST strictly guard against common Text-to-SQL failure modes:

=== PROBLEM 1: ERRONEOUS DATA ANALYSIS PITFALLS ===
1. GRAIN & COUNTING MISMATCH:
   - When calculating rates or user activity, NEVER count row occurrences (COUNT(*), COUNT(e.event_id)) when the metric asks for unique entities.
   - Always count distinct entities: COUNT(DISTINCT u.user_id), COUNT(DISTINCT s.staff_no).

2. FILTER MISPLACEMENT ON OUTER JOINS (JOIN LEAK):
   - Placing a filter on the right-hand table of a LEFT JOIN inside the WHERE clause implicitly converts it into an INNER JOIN, dropping churned/absent members!
   - Keep cohort filters on the left table in WHERE, and evaluate right-table activity conditionally (e.g., inside CASE WHEN or inside the ON clause).

3. SAFE MATH & DIVISION GUARDS:
   - Always guard divisions against zero using NULLIF(denominator, 0).
   - Multiply by 100.0 (float) BEFORE dividing to prevent integer division truncation.
   - Format rates using ROUND(..., 2).

4. MULTI-TABLE CARTESIAN EXPLOSION:
   - When combining aggregates across multiple independent tables (e.g. group_loans and individual_loans payouts), NEVER join them directly before aggregating.
   - Pre-aggregate monthly sums in separate Common Table Expressions (CTEs), then FULL OUTER JOIN the summary CTEs.

=== PROBLEM 2: SCHEMA LINKING & JOIN GRAPH INTEGRITY ===
1. SCHEMA GROUNDING & PRUNING:
   - Always utilize 'prune_schema' or 'get_table_schema' to inspect exact column names. Never hallucinate columns.
   - Ground categorical terms and user typos using 'resolve_categorical_literals' (e.g. "Maharahtra" -> 'Maharashtra', "Tier 1" -> 'T1').

2. JOIN PATH CONTINUITY:
   - Always use verified join predicates from 'find_join_path'.
   - Note physical column mismatches (e.g., individual_loans.outlet_id = outlets.outlet_code).

=== OUTPUT CONSTRAINTS ===
- Emit ONLY a single, read-only SELECT or WITH statement.
- Do NOT include DDL (CREATE, ALTER, DROP) or DML (INSERT, UPDATE, DELETE).
- End your query with a semicolon.
"""


def get_text2sql_prompt_template() -> ChatPromptTemplate:
    """Creates a ready-to-use LangChain ChatPromptTemplate for Text-to-SQL.

    Expected input variables:
    - question: The user's natural language question.
    - schema_context (optional): Pruned compact schema card and join predicates.
    - literal_hints (optional): Discovered categorical literals from resolve_categorical_literals.

    Returns:
        ChatPromptTemplate configured for LangChain LLM invocations.
    """
    human_template = """Question: {question}

{schema_context}

{literal_hints}

Generate the semantically guarded SQL query following the analytical rules."""

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_TEXT2SQL),
        HumanMessagePromptTemplate.from_template(human_template),
    ])


def format_langchain_context(
    pruned_schema_markdown: Optional[str] = None,
    grounded_literals: Optional[List[Dict[str, Any]]] = None,
    join_paths: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Formats schema cards, discovered literals, and join paths for prompt injection."""
    schema_parts: List[str] = []
    if pruned_schema_markdown:
        schema_parts.append(f"### Relevant Database Schema (Pruned via MCP):\n{pruned_schema_markdown}")
    if join_paths:
        paths_str = "\n".join([f"- {p}" for p in join_paths])
        schema_parts.append(f"### Discovered Relational Join Keys:\n{paths_str}")
    schema_context = "\n\n".join(schema_parts) if schema_parts else ""

    literal_parts: List[str] = []
    if grounded_literals:
        literal_parts.append("### Grounded Categorical Literals:")
        for lit in grounded_literals:
            literal_parts.append(
                f"- User term '{lit.get('term')}' maps to database column {lit.get('table_name')}.{lit.get('column_name')} = '{lit.get('database_literal')}'"
            )
    literal_hints = "\n".join(literal_parts) if literal_parts else ""

    return {
        "schema_context": schema_context,
        "literal_hints": literal_hints,
    }
