"""Schema Flattener: Recursively converts nested STRUCT, ARRAY, and JSON types to dot-notation paths."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class FlattenedField(BaseModel):
    """Represents a column or flattened nested subfield path."""

    table_name: str = Field(..., description="Name of the table")
    root_column: str = Field(..., description="Top-level column name in the database table")
    path: str = Field(..., description="Dot-notation path, e.g. event_params.value.string_value")
    data_type: str = Field(..., description="Primitive or leaf data type (e.g. VARCHAR, BIGINT)")
    is_nested: bool = Field(default=False, description="True if this field is inside a STRUCT or ARRAY")
    is_array: bool = Field(default=False, description="True if any ancestor is an ARRAY requiring UNNEST")
    parent_array_path: Optional[str] = Field(default=None, description="Path to the nearest enclosing array")
    unnest_syntax: Optional[str] = Field(default=None, description="Recommended UNNEST syntax expression")
    comment: Optional[str] = Field(default=None, description="Description or semantic comment")

    @property
    def qualified_path(self) -> str:
        """Full table-qualified dot path: table.path"""
        return f"{self.table_name}.{self.path}"


def _split_struct_fields(struct_body: str) -> List[Tuple[str, str]]:
    """Split comma-separated fields inside STRUCT(...) taking nested parentheses and quotes into account."""
    fields: List[Tuple[str, str]] = []
    depth = 0
    in_quote = False
    current: List[str] = []

    for char in struct_body:
        if char == '"' or char == "'":
            in_quote = not in_quote
            current.append(char)
        elif not in_quote:
            if char in "([<":
                depth += 1
                current.append(char)
            elif char in ")]>":
                depth -= 1
                current.append(char)
            elif char == "," and depth == 0:
                chunk = "".join(current).strip()
                if chunk:
                    fields.append(_parse_single_field(chunk))
                current = []
            else:
                current.append(char)
        else:
            current.append(char)

    chunk = "".join(current).strip()
    if chunk:
        fields.append(_parse_single_field(chunk))

    return fields


def _parse_single_field(field_chunk: str) -> Tuple[str, str]:
    """Parse 'name TYPE' or '"name" TYPE' into (name, TYPE)."""
    field_chunk = field_chunk.strip()
    # Check if field name is quoted
    if field_chunk.startswith('"'):
        end_quote = field_chunk.find('"', 1)
        if end_quote != -1:
            name = field_chunk[1:end_quote]
            type_str = field_chunk[end_quote + 1:].strip()
            return name, type_str
    elif field_chunk.startswith("'"):
        end_quote = field_chunk.find("'", 1)
        if end_quote != -1:
            name = field_chunk[1:end_quote]
            type_str = field_chunk[end_quote + 1:].strip()
            return name, type_str

    # Unquoted name: split on first whitespace
    parts = field_chunk.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    elif len(parts) == 1:
        return parts[0].strip(), "UNKNOWN"
    return "unknown", "UNKNOWN"


class SchemaFlattener:
    """Problem 2 Component: Recursively flattens complex nested structures into dot-notation paths."""

    @classmethod
    def parse_type_string(
        cls,
        table_name: str,
        root_column: str,
        current_path: str,
        type_str: str,
        is_array: bool = False,
        parent_array: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> List[FlattenedField]:
        """Recursively parses a SQL/DuckDB data type string (STRUCT, ARRAY, etc.) into FlattenedFields."""
        type_clean = type_str.strip()

        # Check for ARRAY wrapper e.g. STRUCT(...)[] or ARRAY(STRUCT(...))
        if type_clean.endswith("[]"):
            inner_type = type_clean[:-2].strip()
            return cls.parse_type_string(
                table_name=table_name,
                root_column=root_column,
                current_path=current_path,
                type_str=inner_type,
                is_array=True,
                parent_array=current_path,
                comment=comment,
            )

        if type_clean.upper().startswith("ARRAY(") and type_clean.endswith(")"):
            inner_type = type_clean[6:-1].strip()
            return cls.parse_type_string(
                table_name=table_name,
                root_column=root_column,
                current_path=current_path,
                type_str=inner_type,
                is_array=True,
                parent_array=current_path,
                comment=comment,
            )

        # Check for STRUCT(...)
        if type_clean.upper().startswith("STRUCT(") and type_clean.endswith(")"):
            body = type_clean[7:-1].strip()
            fields = _split_struct_fields(body)
            flattened: List[FlattenedField] = []

            for field_name, sub_type in fields:
                sub_path = f"{current_path}.{field_name}" if current_path else field_name
                sub_flattened = cls.parse_type_string(
                    table_name=table_name,
                    root_column=root_column,
                    current_path=sub_path,
                    type_str=sub_type,
                    is_array=is_array,
                    parent_array=parent_array,
                    comment=comment,
                )
                flattened.extend(sub_flattened)
            return flattened

        # Leaf primitive type (VARCHAR, INT, BIGINT, TIMESTAMP, etc.)
        unnest_syntax = f"UNNEST({parent_array})" if is_array and parent_array else None
        return [
            FlattenedField(
                table_name=table_name,
                root_column=root_column,
                path=current_path,
                data_type=type_clean,
                is_nested=("." in current_path),
                is_array=is_array,
                parent_array_path=parent_array,
                unnest_syntax=unnest_syntax,
                comment=comment,
            )
        ]

    @classmethod
    def flatten_dict_schema(
        cls,
        table_name: str,
        schema_dict: Dict[str, Any],
        prefix: str = "",
        root_col: Optional[str] = None,
        is_array: bool = False,
        parent_array: Optional[str] = None,
    ) -> List[FlattenedField]:
        """Flattens a dictionary / JSON representation of nested schemas."""
        flattened: List[FlattenedField] = []

        for key, val in schema_dict.items():
            current_path = f"{prefix}.{key}" if prefix else key
            current_root = root_col or key

            if isinstance(val, dict):
                # Struct / Object
                flattened.extend(
                    cls.flatten_dict_schema(
                        table_name=table_name,
                        schema_dict=val,
                        prefix=current_path,
                        root_col=current_root,
                        is_array=is_array,
                        parent_array=parent_array,
                    )
                )
            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                # Array of Structs
                flattened.extend(
                    cls.flatten_dict_schema(
                        table_name=table_name,
                        schema_dict=val[0],
                        prefix=current_path,
                        root_col=current_root,
                        is_array=True,
                        parent_array=current_path,
                    )
                )
            elif isinstance(val, list):
                # Array of primitives
                elem_type = str(val[0]) if len(val) > 0 else "VARCHAR"
                flattened.append(
                    FlattenedField(
                        table_name=table_name,
                        root_column=current_root,
                        path=current_path,
                        data_type=f"ARRAY<{elem_type}>",
                        is_nested=True,
                        is_array=True,
                        parent_array_path=current_path,
                        unnest_syntax=f"UNNEST({current_path})",
                    )
                )
            else:
                # Leaf string type
                flattened.append(
                    FlattenedField(
                        table_name=table_name,
                        root_column=current_root,
                        path=current_path,
                        data_type=str(val),
                        is_nested=("." in current_path),
                        is_array=is_array,
                        parent_array_path=parent_array,
                        unnest_syntax=f"UNNEST({parent_array})" if is_array and parent_array else None,
                    )
                )

        return flattened

    @classmethod
    def flatten_duckdb_table(cls, con: Any, table_name: str) -> List[FlattenedField]:
        """Inspects and flattens all flat and nested columns for a DuckDB table."""
        describe_rows = con.execute(f"DESCRIBE {table_name};").fetchall()
        flattened: List[FlattenedField] = []

        for row in describe_rows:
            col_name = str(row[0])
            col_type = str(row[1])
            fields = cls.parse_type_string(
                table_name=table_name,
                root_column=col_name,
                current_path=col_name,
                type_str=col_type,
            )
            flattened.extend(fields)

        return flattened

    @classmethod
    def flatten_postgres_table(
        cls,
        con: Any,
        table_name: str,
        schema_name: str = "public",
    ) -> List[FlattenedField]:
        """Inspects and flattens columns for a PostgreSQL table, including ARRAY, JSON, and JSONB."""
        query = """
        SELECT
            c.column_name,
            c.data_type,
            c.udt_name,
            d.description AS column_comment
        FROM information_schema.columns c
        LEFT JOIN pg_class cl ON cl.relname = c.table_name
        LEFT JOIN pg_namespace ns ON ns.oid = cl.relnamespace AND ns.nspname = c.table_schema
        LEFT JOIN pg_description d ON d.objoid = cl.oid AND d.objsubid = c.ordinal_position
        WHERE c.table_name = %s
          AND c.table_schema = %s
        ORDER BY c.ordinal_position;
        """
        flattened: List[FlattenedField] = []
        try:
            with con.cursor() as cur:
                cur.execute(query, (table_name, schema_name))
                rows = cur.fetchall()
        except Exception:
            fallback_query = """
            SELECT column_name, data_type, udt_name, NULL AS column_comment
            FROM information_schema.columns
            WHERE table_name = %s
              AND table_schema = %s
            ORDER BY ordinal_position;
            """
            try:
                con.rollback()
            except Exception:
                pass
            with con.cursor() as cur:
                cur.execute(fallback_query, (table_name, schema_name))
                rows = cur.fetchall()

        for row in rows:
            if not row or len(row) < 2:
                continue
            col_name = str(row[0])
            data_type = str(row[1]).upper()
            udt_name = str(row[2]) if len(row) > 2 else data_type.lower()
            comment = str(row[3]) if len(row) > 3 and row[3] else None

            if data_type == "ARRAY" or udt_name.startswith("_"):
                elem_type = udt_name.lstrip("_").upper()
                type_str = f"ARRAY<{elem_type}>"
                flattened.append(
                    FlattenedField(
                        table_name=table_name,
                        root_column=col_name,
                        path=col_name,
                        data_type=type_str,
                        is_nested=True,
                        is_array=True,
                        parent_array_path=col_name,
                        unnest_syntax=f"UNNEST({col_name})",
                        comment=comment,
                    )
                )
            elif data_type in ("JSON", "JSONB"):
                flattened.append(
                    FlattenedField(
                        table_name=table_name,
                        root_column=col_name,
                        path=col_name,
                        data_type=data_type,
                        is_nested=True,
                        is_array=False,
                        parent_array_path=None,
                        unnest_syntax=None,
                        comment=comment or f"{data_type} document field",
                    )
                )
            else:
                flattened.append(
                    FlattenedField(
                        table_name=table_name,
                        root_column=col_name,
                        path=col_name,
                        data_type=data_type,
                        is_nested=False,
                        is_array=False,
                        parent_array_path=None,
                        unnest_syntax=None,
                        comment=comment,
                    )
                )

        return flattened

    @classmethod
    def generate_schema_card(cls, table_name: str, fields: List[FlattenedField], description: str = "") -> str:
        """Generates an LLM-friendly schema card with dot paths and nested hints."""
        lines = [f"Table: {table_name}"]
        if description:
            lines.append(f"  Description: {description}")
        lines.append("  Columns / Fields:")

        for f in fields:
            flags = []
            if f.is_array:
                flags.append("ARRAY (requires UNNEST)")
            if f.is_nested:
                flags.append("NESTED")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            desc = f" -- {f.comment}" if f.comment else ""
            lines.append(f"    - {f.path} ({f.data_type}){flag_str}{desc}")

        return "\n".join(lines)

    @classmethod
    def get_dialect_unnest_expression(cls, field: FlattenedField, dialect: str) -> Optional[str]:
        """Returns the dialect-specific array unnesting / lateral flatten SQL clause.

        Architectural Note:
            Schema flattening produces dot-notation paths and identifies array unnest requirements.
            Flattening a schema alone does NOT automatically execute dialect-specific array unnesting.
            Dialect-specific SQL generation requires explicit syntax transformations handled here
            or within dialect-specific query planning.

        Args:
            field: FlattenedField with is_array=True.
            dialect: Target SQL dialect ('duckdb', 'bigquery', 'snowflake', 'postgres').

        Returns:
            SQL fragment string for unnesting, or None if the field is not an array.
        """
        if not field.is_array or not field.parent_array_path:
            return None

        d = dialect.lower().strip()
        arr_path = field.parent_array_path
        alias = field.root_column

        if d in ("duckdb", "sqlite"):
            return f"UNNEST({arr_path}) AS t({alias})"
        elif d == "bigquery":
            return f"CROSS JOIN UNNEST({arr_path}) AS {alias}"
        elif d == "snowflake":
            return f", LATERAL FLATTEN(input => {arr_path}) AS {alias}"
        elif d in ("postgres", "postgresql"):
            return f"CROSS JOIN json_array_elements({arr_path}) AS {alias}"
        else:
            return f"UNNEST({arr_path}) AS {alias}"

