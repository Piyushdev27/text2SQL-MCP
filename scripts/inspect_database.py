"""Database inspection script.

Connects to PostgreSQL, lists all accessible tables, inspects column names,
data types, nullability, primary keys, foreign keys, and row counts.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Tuple
from src.config import get_settings
from src.db import get_db_cursor, test_connection


def inspect_postgres() -> None:
    """Performs full schema inspection of the target PostgreSQL database."""
    settings = get_settings()
    print("=" * 70)
    print("PostgreSQL Database Schema Inspector")
    print("=" * 70)

    try:
        conn_info = test_connection(settings)
        print(f"[OK] Connected to: {conn_info['database']} on {conn_info['host']}:{conn_info['port']}")
        print(f"     User: {conn_info['user']}")
        print(f"     Version: {conn_info['version']}")
        print(f"     Target Schema: {settings.postgres_schema}")
    except Exception as exc:
        print(f"[ERROR] Failed to connect to PostgreSQL: {exc}")
        sys.exit(1)

    with get_db_cursor(settings) as cur:
        # 1. Fetch all accessible base tables
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name;
            """,
            (settings.postgres_schema,),
        )
        tables = [r[0] for r in cur.fetchall()]
        print(f"\nDiscovered {len(tables)} tables in schema '{settings.postgres_schema}':")

        # 2. Fetch primary keys
        cur.execute(
            """
            SELECT kcu.table_name, kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
              AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = %s;
            """,
            (settings.postgres_schema,),
        )
        pks: Dict[str, List[str]] = {}
        for tbl, col in cur.fetchall():
            pks.setdefault(tbl, []).append(col)

        # 3. For each table, inspect columns and row counts
        for tbl in tables:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (settings.postgres_schema, tbl),
            )
            columns = cur.fetchall()

            # Count rows
            cur.execute(f'SELECT count(*) FROM "{settings.postgres_schema}"."{tbl}";')
            row_count = cur.fetchone()[0]

            pk_list = pks.get(tbl, [])
            pk_str = f" [PK: {', '.join(pk_list)}]" if pk_list else ""

            print(f"\n* Table: {tbl} ({row_count:,} rows){pk_str}")
            for col_name, data_type, is_nullable in columns:
                nullable_flag = "NULL" if is_nullable == "YES" else "NOT NULL"
                is_pk = " (PK)" if col_name in pk_list else ""
                print(f"    - {col_name:<24} {data_type:<28} {nullable_flag}{is_pk}")

    print("\n" + "=" * 70)
    print("Database inspection completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    inspect_postgres()
