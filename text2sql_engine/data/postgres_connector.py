"""PostgreSQL connector, schema introspector, and query execution helper."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from text2sql_engine.core.config import get_postgres_connection, get_settings
from text2sql_engine.core.exceptions import ConfigurationError, ExecutionError
from text2sql_engine.modules.problem2_retriever.schema_flattener import FlattenedField, SchemaFlattener

logger = logging.getLogger(__name__)


class PostgresConnector:
    """Manages live PostgreSQL connections, schema introspection, and safe query execution."""

    def __init__(
        self,
        connection: Optional[Any] = None,
        uri: Optional[str] = None,
        schema_name: str = "public",
    ):
        self.schema_name = schema_name
        if connection is not None:
            self.connection = connection
            self._owns_connection = False
        else:
            self.connection = get_postgres_connection(uri=uri)
            self._owns_connection = True

    def close(self) -> None:
        """Closes the connection if owned by this instance."""
        if self._owns_connection and self.connection and not self.connection.closed:
            try:
                self.connection.close()
            except Exception:
                pass

    def get_table_names(self) -> List[str]:
        """Returns all base table names in the target schema."""
        query = """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
        with self.connection.cursor() as cur:
            cur.execute(query, (self.schema_name,))
            rows = cur.fetchall()
            return [r[0] for r in rows]

    def get_table_comments(self) -> Dict[str, str]:
        """Returns a mapping of table_name -> description from pg_description."""
        query = """
        SELECT
            c.relname AS table_name,
            d.description AS table_comment
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_description d ON d.objoid = c.oid AND d.objsubid = 0
        WHERE n.nspname = %s
          AND c.relkind = 'r';
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(query, (self.schema_name,))
                return {row[0]: row[1] for row in cur.fetchall() if row[1]}
        except Exception:
            return {}

    def get_primary_keys(self) -> Dict[str, List[str]]:
        """Returns a mapping of table_name -> list of primary key column names."""
        query = """
        SELECT
            kcu.table_name,
            kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
          AND tc.table_schema = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = %s
        ORDER BY kcu.table_name, kcu.ordinal_position;
        """
        pks: Dict[str, List[str]] = {}
        try:
            with self.connection.cursor() as cur:
                cur.execute(query, (self.schema_name,))
                for table_name, col in cur.fetchall():
                    pks.setdefault(table_name, []).append(col)
        except Exception:
            pass
        return pks

    def get_foreign_keys(self) -> Dict[str, List[str]]:
        """Returns a mapping of table_name -> list of foreign key column references."""
        query = """
        SELECT
            kcu.table_name,
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
          AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
          AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = %s;
        """
        fks: Dict[str, List[str]] = {}
        try:
            with self.connection.cursor() as cur:
                cur.execute(query, (self.schema_name,))
                for table_name, col, f_table, f_col in cur.fetchall():
                    fks.setdefault(table_name, []).append(f"{col}->{f_table}.{f_col}")
        except Exception:
            pass
        return fks

    def flatten_table(self, table_name: str) -> List[FlattenedField]:
        """Inspects and returns flattened column fields for a PostgreSQL table."""
        return SchemaFlattener.flatten_postgres_table(
            self.connection,
            table_name=table_name,
            schema_name=self.schema_name,
        )

    def sample_distinct_values(
        self,
        table_name: str,
        column_name: str,
        limit: int = 15,
    ) -> List[Tuple[str, str]]:
        """Samples up to `limit` distinct values for a column to populate the DistinctValueIndexer."""
        # Sanity check identifier names
        safe_table = table_name.replace('"', '""')
        safe_col = column_name.replace('"', '""')
        safe_schema = self.schema_name.replace('"', '""')

        query = f"""
        SELECT DISTINCT "{safe_col}"
        FROM "{safe_schema}"."{safe_table}"
        WHERE "{safe_col}" IS NOT NULL
        LIMIT %s;
        """
        try:
            with self.connection.cursor() as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()
                results: List[Tuple[str, str]] = []
                for (val,) in rows:
                    str_val = str(val).strip()
                    if str_val:
                        # (raw_literal, semantic_label)
                        results.append((str_val, str_val))
                return results
        except Exception:
            self.connection.rollback()
            return []

    def inspect_database_schema(
        self,
        tables_to_include: Optional[List[str]] = None,
        sample_values_columns: Optional[Dict[str, List[str]]] = None,
        max_distinct_sample: int = 15,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, List[Tuple[str, str]]]]]:
        """Introspects tables, columns, constraints, and categorical literals for the vector retriever.

        Returns:
            tables_metadata: Dict[table_name, {description, fields, primary_keys, foreign_keys}]
            sample_values: Dict[table_name, Dict[column_name, List[(literal, label)]]]
        """
        all_tables = self.get_table_names()
        target_tables = [t for t in all_tables if (not tables_to_include or t in tables_to_include)]

        comments = self.get_table_comments()
        pks = self.get_primary_keys()
        fks = self.get_foreign_keys()

        tables_meta: Dict[str, Dict[str, Any]] = {}
        sample_values: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}

        for tbl in target_tables:
            fields = self.flatten_table(tbl)
            desc = comments.get(tbl, f"PostgreSQL table {tbl}")
            tables_meta[tbl] = {
                "description": desc,
                "fields": fields,
                "primary_keys": pks.get(tbl, []),
                "foreign_keys": fks.get(tbl, []),
            }

            # Sample values for low-cardinality text/character columns
            sample_values[tbl] = {}
            specified_cols = sample_values_columns.get(tbl) if sample_values_columns else None

            for field in fields:
                should_sample = False
                if specified_cols is not None:
                    should_sample = field.root_column in specified_cols
                else:
                    # Auto-detect text/varchar/enum columns with short names often used as categories
                    upper_type = field.data_type.upper()
                    if any(t in upper_type for t in ("VARCHAR", "TEXT", "CHAR", "USER-DEFINED")) and not field.is_nested:
                        col_lower = field.root_column.lower()
                        if any(term in col_lower for term in ("status", "code", "type", "category", "region", "country", "state")):
                            should_sample = True

                if should_sample:
                    vals = self.sample_distinct_values(tbl, field.root_column, limit=max_distinct_sample)
                    if vals:
                        sample_values[tbl][field.root_column] = vals

        return tables_meta, sample_values

    def execute_query(self, sql: str) -> List[Dict[str, Any]]:
        """Executes a SQL query against PostgreSQL and returns results as a list of dicts."""
        try:
            with self.connection.cursor() as cur:
                cur.execute(sql)
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchall()
                    results: List[Dict[str, Any]] = []
                    for row in rows:
                        results.append(dict(zip(columns, row)))
                    return results
                else:
                    return []
        except Exception as e:
            try:
                self.connection.rollback()
            except Exception:
                pass
            raise ExecutionError(f"PostgreSQL execution error: {e}") from e
