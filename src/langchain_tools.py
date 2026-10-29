"""LangChain Tool Adapters for Enterprise Text-to-SQL.

Exposes all 9 Model Context Protocol (MCP) tools as native LangChain BaseTool
instances (StructuredTool), supporting both synchronous (.invoke()) and
asynchronous (.ainvoke()) execution with typed Pydantic input schemas.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from src.query_executor import MCPQueryExecutor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed Input Schemas for LangChain Tool Calling
# ---------------------------------------------------------------------------

class ListTablesInput(BaseModel):
    """Input for listing database tables."""
    target: str = Field(
        default="postgres",
        description="Target database: 'postgres' (live warehouse) or 'local' (benchmark SQLite).",
    )


class GetTableSchemaInput(BaseModel):
    """Input for retrieving schema metadata of a specific table."""
    table_name: str = Field(
        description="Name of the table to inspect (e.g., 'outlets', 'staff', 'group_loans', 'individual_loans')."
    )
    target: str = Field(
        default="postgres",
        description="Target database: 'postgres' or 'local'.",
    )


class GetSampleRowsInput(BaseModel):
    """Input for fetching sample rows to inspect realistic column values."""
    table_name: str = Field(
        description="Name of the table to sample from."
    )
    limit: int = Field(
        default=3,
        description="Number of sample rows to fetch (clamped between 1 and 10).",
    )
    target: str = Field(
        default="postgres",
        description="Target database: 'postgres' or 'local'.",
    )


class ValidateSqlInput(BaseModel):
    """Input for AST security validation of an SQL query."""
    sql: str = Field(
        description="SQL query string to validate using SQLGlot AST inspection."
    )
    target: str = Field(
        default="postgres",
        description="Target database: 'postgres' or 'local'.",
    )


class ExecuteReadonlySqlInput(BaseModel):
    """Input for executing a safe read-only SQL query."""
    sql: str = Field(
        description="Read-only SELECT/WITH query string to execute. Modifying statements (INSERT, UPDATE, DELETE, DROP) will be blocked."
    )
    target: str = Field(
        default="postgres",
        description="Target database: 'postgres' or 'local'.",
    )


class ResolveLiteralsInput(BaseModel):
    """Input for grounding categorical terms and fixing typos."""
    natural_language_query: str = Field(
        description="User question containing potential categorical terms, abbreviations, or typos (e.g. 'Maharahtra', 'Tier 1', 'PR')."
    )
    target: str = Field(
        default="postgres",
        description="Target database: 'postgres' or 'local'.",
    )


class PruneSchemaInput(BaseModel):
    """Input for Problem 2 coarse table and fine column pruning."""
    natural_language_query: str = Field(
        description="User question to link and prune schema against."
    )
    target: str = Field(
        default="postgres",
        description="Target database: 'postgres' or 'local'.",
    )
    top_k_tables: int = Field(
        default=3,
        description="Maximum number of candidate tables to select (defaults to 3).",
    )


class FindJoinPathInput(BaseModel):
    """Input for relational knowledge graph shortest join path search."""
    source_table: str = Field(
        description="Originating table name (e.g. 'staff_attendance')."
    )
    target_table: str = Field(
        description="Target referenced table name (e.g. 'outlets')."
    )
    target: str = Field(
        default="postgres",
        description="Target database: 'postgres' or 'local'.",
    )


class GetSchemaGraphInput(BaseModel):
    """Input for retrieving the complete relational knowledge graph."""
    target: str = Field(
        default="postgres",
        description="Target database: 'postgres' or 'local'.",
    )


# ---------------------------------------------------------------------------
# LangChain Tool Factory
# ---------------------------------------------------------------------------

def get_langchain_tools(
    executor: Optional[MCPQueryExecutor] = None,
    default_target: str = "postgres",
) -> List[BaseTool]:
    """Builds and returns all 9 MCP tools as native LangChain BaseTool instances.

    Each tool is configured with:
    - Typed Pydantic args_schema for LLM function calling
    - Synchronous handler (.invoke())
    - Asynchronous handler (.ainvoke())
    - Descriptive operational guidelines preventing SQL hallucinations

    Args:
        executor: Optional MCPQueryExecutor instance. Creates a new instance if None.
        default_target: Default database target ('postgres' or 'local').

    Returns:
        List of 9 LangChain BaseTool objects.
    """
    exec_inst = executor or MCPQueryExecutor()

    # Tool 1: list_tables
    def _list_tables(target: str = default_target) -> Dict[str, Any]:
        return exec_inst.list_tables_sync(target=target)

    async def _alist_tables(target: str = default_target) -> Dict[str, Any]:
        return await exec_inst.list_tables(target=target)

    list_tables_tool = StructuredTool.from_function(
        func=_list_tables,
        coroutine=_alist_tables,
        name="list_tables",
        description=(
            "Lists all accessible tables in the target database. "
            "Use this tool when beginning a query to discover available entities."
        ),
        args_schema=ListTablesInput,
    )

    # Tool 2: get_table_schema
    def _get_table_schema(table_name: str, target: str = default_target) -> Dict[str, Any]:
        return exec_inst.get_table_schema_sync(table_name=table_name, target=target)

    async def _aget_table_schema(table_name: str, target: str = default_target) -> Dict[str, Any]:
        return await exec_inst.get_table_schema(table_name=table_name, target=target)

    get_table_schema_tool = StructuredTool.from_function(
        func=_get_table_schema,
        coroutine=_aget_table_schema,
        name="get_table_schema",
        description=(
            "Retrieves full schema metadata for a database table, including column names, "
            "data types, nullability, and primary key status."
        ),
        args_schema=GetTableSchemaInput,
    )

    # Tool 3: get_sample_rows
    def _get_sample_rows(table_name: str, limit: int = 3, target: str = default_target) -> Dict[str, Any]:
        return exec_inst.get_sample_rows_sync(table_name=table_name, limit=limit, target=target)

    async def _aget_sample_rows(table_name: str, limit: int = 3, target: str = default_target) -> Dict[str, Any]:
        return await exec_inst.get_sample_rows(table_name=table_name, limit=limit, target=target)

    get_sample_rows_tool = StructuredTool.from_function(
        func=_get_sample_rows,
        coroutine=_aget_sample_rows,
        name="get_sample_rows",
        description=(
            "Fetches bounded sample rows (max 10) from a table. "
            "Essential for inspecting realistic values, date formats, status strings, and codes."
        ),
        args_schema=GetSampleRowsInput,
    )

    # Tool 4: validate_sql
    def _validate_sql(sql: str, target: str = default_target) -> Dict[str, Any]:
        return exec_inst.validate_sql_sync(sql=sql, dialect=target)

    async def _avalidate_sql(sql: str, target: str = default_target) -> Dict[str, Any]:
        return await exec_inst.validate_sql(sql=sql, dialect=target)

    validate_sql_tool = StructuredTool.from_function(
        func=_validate_sql,
        coroutine=_avalidate_sql,
        name="validate_sql",
        description=(
            "Validates SQL syntax and safety using AST inspection. "
            "Ensures the query is a single read-only SELECT or WITH statement. Blocks DDL/DML."
        ),
        args_schema=ValidateSqlInput,
    )

    # Tool 5: execute_readonly_sql
    def _execute_readonly_sql(sql: str, target: str = default_target) -> Dict[str, Any]:
        return exec_inst.execute_readonly_sql_sync(sql=sql, target=target)

    async def _aexecute_readonly_sql(sql: str, target: str = default_target) -> Dict[str, Any]:
        return await exec_inst.execute_readonly_sql(sql=sql, target=target)

    execute_readonly_sql_tool = StructuredTool.from_function(
        func=_execute_readonly_sql,
        coroutine=_aexecute_readonly_sql,
        name="execute_readonly_sql",
        description=(
            "Executes a validated read-only SQL query against the database with strict 60s timeout "
            "and row limit protection. Returns columns, rows, and row count."
        ),
        args_schema=ExecuteReadonlySqlInput,
    )

    # Tool 6: resolve_categorical_literals
    def _resolve_literals(natural_language_query: str, target: str = default_target) -> List[Dict[str, Any]]:
        return exec_inst.resolve_categorical_literals_sync(user_query=natural_language_query, target=target)

    async def _aresolve_literals(natural_language_query: str, target: str = default_target) -> List[Dict[str, Any]]:
        return await exec_inst.resolve_categorical_literals(user_query=natural_language_query, target=target)

    resolve_literals_tool = StructuredTool.from_function(
        func=_resolve_literals,
        coroutine=_aresolve_literals,
        name="resolve_categorical_literals",
        description=(
            "Grounds natural language terms and repairs typos (e.g. 'Maharahtra' -> 'Maharashtra', "
            "'Tier 1' -> 'T1', 'casual leave' -> 'CL') against live database values. "
            "Always call this tool before generating WHERE clauses."
        ),
        args_schema=ResolveLiteralsInput,
    )

    # Tool 7: prune_schema (Problem 2)
    def _prune_schema(natural_language_query: str, target: str = default_target, top_k_tables: int = 3) -> Dict[str, Any]:
        return exec_inst.prune_schema_sync(user_query=natural_language_query, target=target, top_k_tables=top_k_tables)

    async def _aprune_schema(natural_language_query: str, target: str = default_target, top_k_tables: int = 3) -> Dict[str, Any]:
        return await exec_inst.prune_schema(user_query=natural_language_query, target=target, top_k_tables=top_k_tables)

    prune_schema_tool = StructuredTool.from_function(
        func=_prune_schema,
        coroutine=_aprune_schema,
        name="prune_schema",
        description=(
            "Problem 2 Schema Linking: Performs coarse table pruning and fine column pruning. "
            "Returns relevant candidate tables, drops unused audit columns (60-85% token reduction), "
            "and generates compact markdown schema cards to eliminate LLM hallucinations."
        ),
        args_schema=PruneSchemaInput,
    )

    # Tool 8: find_join_path (Problem 2)
    def _find_join_path(source_table: str, target_table: str, target: str = default_target) -> Dict[str, Any]:
        return exec_inst.find_join_path_sync(source_table=source_table, target_table=target_table, target=target)

    async def _afind_join_path(source_table: str, target_table: str, target: str = default_target) -> Dict[str, Any]:
        return await exec_inst.find_join_path(source_table=source_table, target_table=target_table, target=target)

    find_join_path_tool = StructuredTool.from_function(
        func=_find_join_path,
        coroutine=_afind_join_path,
        name="find_join_path",
        description=(
            "Problem 2 Relational Graph: Finds the shortest join path between two tables using BFS graph traversal. "
            "Discovers single-hop and multi-hop ON predicates (e.g. staff_attendance -> staff -> outlets). "
            "Use this whenever a query joins two or more tables."
        ),
        args_schema=FindJoinPathInput,
    )

    # Tool 9: get_schema_graph (Problem 2)
    def _get_schema_graph(target: str = default_target) -> Dict[str, Any]:
        return exec_inst.get_schema_graph_sync(target=target)

    async def _aget_schema_graph(target: str = default_target) -> Dict[str, Any]:
        return await exec_inst.get_schema_graph(target=target)

    get_schema_graph_tool = StructuredTool.from_function(
        func=_get_schema_graph,
        coroutine=_aget_schema_graph,
        name="get_schema_graph",
        description=(
            "Returns the complete logical relational knowledge graph including table descriptions, "
            "logical primary keys, domain tags, and relational join edges."
        ),
        args_schema=GetSchemaGraphInput,
    )

    return [
        list_tables_tool,
        get_table_schema_tool,
        get_sample_rows_tool,
        validate_sql_tool,
        execute_readonly_sql_tool,
        resolve_literals_tool,
        prune_schema_tool,
        find_join_path_tool,
        get_schema_graph_tool,
    ]


def get_langchain_tool_map(
    executor: Optional[MCPQueryExecutor] = None,
    default_target: str = "postgres",
) -> Dict[str, BaseTool]:
    """Returns a dictionary mapping tool name -> LangChain BaseTool."""
    tools = get_langchain_tools(executor=executor, default_target=default_target)
    return {tool.name: tool for tool in tools}
