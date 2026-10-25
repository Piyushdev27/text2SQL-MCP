"""Schema inspector module for extracting database metadata and schema cards.

Supports both live PostgreSQL and local SQLite databases. Extracts table names,
column definitions, types, primary keys, and sample rows formatted for LLM consumption.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from src.config import Settings, get_settings
from src.db import execute_readonly_query, get_db_cursor, get_local_db_connection

logger = logging.getLogger(__name__)


class SchemaInspector:
    """Introspects relational database schemas and formats schema context for Text-to-SQL."""

    def __init__(self, target: str = "postgres", settings: Optional[Settings] = None):
        """Initializes inspector for either 'postgres' or 'local' database."""
        self.target = target.lower()
        self.settings = settings or get_settings()

    def get_table_names(self) -> List[str]:
        """Returns sorted list of permitted base table names."""
        if self.target == "local":
            conn = get_local_db_connection(self.settings)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
            tables = [row["name"] for row in cur.fetchall()]
            conn.close()
            return tables

        # PostgreSQL
        with get_db_cursor(self.settings) as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """,
                (self.settings.postgres_schema,),
            )
            return [row[0] for row in cur.fetchall()]

    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """Returns detailed column specifications for a specific table."""
        # Sanitize table_name against permitted tables to avoid SQL injection
        permitted = self.get_table_names()
        if table_name not in permitted:
            raise ValueError(f"Table '{table_name}' is not in permitted tables: {permitted}")

        if self.target == "local":
            conn = get_local_db_connection(self.settings)
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info('{table_name}');")
            cols = []
            for row in cur.fetchall():
                cols.append({
                    "name": row["name"],
                    "data_type": row["type"].upper(),
                    "nullable": not bool(row["notnull"]),
                    "is_primary_key": bool(row["pk"]),
                })
            conn.close()
            return cols

        # PostgreSQL
        with get_db_cursor(self.settings) as cur:
            # Fetch primary keys
            cur.execute(
                """
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                  AND tc.table_schema = kcu.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = %s
                  AND tc.table_name = %s;
                """,
                (self.settings.postgres_schema, table_name),
            )
            pks = {row[0] for row in cur.fetchall()}

            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (self.settings.postgres_schema, table_name),
            )
            columns = []
            for col_name, data_type, is_nullable in cur.fetchall():
                columns.append({
                    "name": col_name,
                    "data_type": data_type.upper(),
                    "nullable": is_nullable == "YES",
                    "is_primary_key": col_name in pks,
                })
            return columns

    def get_schema_card(self, table_name: str) -> str:
        """Generates an LLM-friendly schema card in Markdown format."""
        columns = self.get_table_columns(table_name)
        lines = [f"### Table: `{table_name}`", "| Column | Type | Nullable | Primary Key |", "|---|---|---|---|"]
        for col in columns:
            nullable_str = "YES" if col["nullable"] else "NO"
            pk_str = "PRIMARY KEY" if col["is_primary_key"] else "-"
            lines.append(f"| `{col['name']}` | `{col['data_type']}` | {nullable_str} | {pk_str} |")
        return "\n".join(lines)

    def get_all_schema_cards(self) -> str:
        """Returns concatenated schema cards for all permitted tables."""
        tables = self.get_table_names()
        cards = [self.get_schema_card(t) for t in tables]
        return "\n\n".join(cards)

    def get_sample_rows(self, table_name: str, limit: int = 3) -> Dict[str, Any]:
        """Safely fetches a bounded sample of rows for a permitted table."""
        permitted = self.get_table_names()
        if table_name not in permitted:
            raise ValueError(f"Table '{table_name}' is not permitted.")

        clamped_limit = min(max(1, limit), 5)  # Enforce 1 <= limit <= 5

        if self.target == "local":
            conn = get_local_db_connection(self.settings)
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM \"{table_name}\" LIMIT {clamped_limit};")
            rows = [dict(r) for r in cur.fetchall()]
            columns = list(rows[0].keys()) if rows else []
            conn.close()
            return {"columns": columns, "rows": rows, "count": len(rows)}

        # PostgreSQL
        query = f'SELECT * FROM "{self.settings.postgres_schema}"."{table_name}" LIMIT {clamped_limit};'
        res = execute_readonly_query(query, settings=self.settings)
        return {"columns": res["columns"], "rows": res["rows"], "count": res["row_count"]}
