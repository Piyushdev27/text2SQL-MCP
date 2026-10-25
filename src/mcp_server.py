"""Official Model Context Protocol (MCP) Server for safe database interaction.

Uses the official Python MCP SDK (MCPServer).
Exposes 5 core tools to LLM applications:
1. list_tables: Lists permitted base tables.
2. get_table_schema: Retrieves formatted table schema cards with types and keys.
3. get_sample_rows: Retrieves a small, bounded sample of records (1-5 rows).
4. validate_sql: Evaluates query AST for syntax errors, multiple statements, and write operations.
5. execute_readonly_sql: Executes approved read-only queries with timeouts and row limits.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from mcp.server.mcpserver import MCPServer

from src.config import Settings, get_settings
from src.db import execute_readonly_query, get_local_db_connection
from src.schema_inspector import SchemaInspector
from src.sql_validator import validate_readonly_sql

logger = logging.getLogger(__name__)

# Initialize official MCP Server
server = MCPServer("postgres_text2sql_server")


@server.tool(
    name="list_tables",
    description="Lists all permitted base tables in the target database ('postgres' or 'local').",
)
def list_tables(target: str = "postgres") -> List[str]:
    """Retrieves sorted list of table names in the active schema."""
    inspector = SchemaInspector(target=target)
    return inspector.get_table_names()


@server.tool(
    name="get_table_schema",
    description="Retrieves column definitions, data types, nullability, and primary keys for a table.",
)
def get_table_schema(table_name: str, target: str = "postgres") -> str:
    """Returns an LLM-friendly schema card in Markdown for the specified table."""
    inspector = SchemaInspector(target=target)
    return inspector.get_schema_card(table_name)


@server.tool(
    name="get_sample_rows",
    description="Retrieves a small, bounded sample of rows (1 to 5 rows) from a permitted table.",
)
def get_sample_rows(table_name: str, limit: int = 3, target: str = "postgres") -> Dict[str, Any]:
    """Fetches sample rows to help the LLM understand cell formats and category values."""
    inspector = SchemaInspector(target=target)
    return inspector.get_sample_rows(table_name, limit=limit)


@server.tool(
    name="validate_sql",
    description="Validates that a SQL query is single-statement, syntactically valid, and strictly read-only.",
)
def validate_sql(sql: str, dialect: str = "postgres") -> Dict[str, Any]:
    """Parses SQL AST using SQLGlot to detect DDL/DML, multi-statements, and dangerous functions."""
    report = validate_readonly_sql(sql, dialect=dialect)
    return report.model_dump()


@server.tool(
    name="execute_readonly_sql",
    description="Executes a verified read-only SQL query with a timeout and returns bounded rows.",
)
def execute_readonly_sql(
    sql: str,
    max_rows: int = 100,
    target: str = "postgres",
) -> Dict[str, Any]:
    """Validates and executes a query, returning columns, rows, and row counts.
    
    Security Guarantee:
    - Queries failing validate_readonly_sql() are rejected immediately.
    - Read-only transaction mode and statement timeouts are enforced at database level.
    """
    dialect = "sqlite" if target.lower() == "local" else "postgres"

    # Step 1: Pre-execution AST validation
    validation = validate_readonly_sql(sql, dialect=dialect)
    if not validation.is_valid:
        return {
            "success": False,
            "error": f"Validation Rejected: {validation.error}",
            "columns": [],
            "rows": [],
            "row_count": 0,
        }

    settings = get_settings()
    bounded_limit = min(max(1, max_rows), settings.max_row_limit)

    # Step 2: Safe execution
    try:
        if target.lower() == "local":
            conn = get_local_db_connection(settings)
            cur = conn.cursor()
            cur.execute(sql)
            rows = [dict(r) for r in cur.fetchmany(bounded_limit)]
            columns = list(rows[0].keys()) if rows else []
            conn.close()
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "target": "local",
            }

        # PostgreSQL execution
        result = execute_readonly_query(sql, max_rows=bounded_limit, settings=settings)
        return {
            "success": True,
            "columns": result["columns"],
            "rows": result["rows"],
            "row_count": result["row_count"],
            "truncated": result["truncated"],
            "target": "postgres",
        }
    except Exception as exc:
        logger.error("Error executing read-only SQL via MCP: %s", exc)
        return {
            "success": False,
            "error": f"Execution error: {exc}",
            "columns": [],
            "rows": [],
            "row_count": 0,
        }


def run_standalone_server() -> None:
    """Entrypoint to run the MCP server using standard I/O (stdio) transport."""
    server.run()


if __name__ == "__main__":
    run_standalone_server()
