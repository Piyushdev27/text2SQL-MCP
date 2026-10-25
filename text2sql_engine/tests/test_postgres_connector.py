"""Unit and integration tests for PostgreSQL connector, schema introspection, and execution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from text2sql_engine.core.config import EngineSettings, reset_settings
from text2sql_engine.core.exceptions import ExecutionError
from text2sql_engine.data.postgres_connector import PostgresConnector
from text2sql_engine.modules.problem2_retriever.schema_flattener import FlattenedField, SchemaFlattener
from text2sql_engine.main import Text2SQLPipeline


class MockCursor:
    """Mock cursor for database introspection and query execution."""

    def __init__(self, fetchall_results=None, description=None):
        self._fetchall_results = fetchall_results or []
        self.description = description
        self.executed_queries = []

    def execute(self, query, params=None):
        self.executed_queries.append((query, params))

    def fetchall(self):
        return self._fetchall_results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class MockConnection:
    """Mock psycopg2 connection."""

    def __init__(self, cursor_factory=None):
        self.cursor_factory = cursor_factory or (lambda: MockCursor())
        self.closed = False
        self.rollback_called = False

    def cursor(self):
        return self.cursor_factory()

    def rollback(self):
        self.rollback_called = True

    def close(self):
        self.closed = True


def test_schema_flattener_postgres_table():
    """Tests flattening PostgreSQL tables with primitive, ARRAY, and JSON/JSONB columns."""
    sample_columns = [
        ("id", "bigint", "int8", "Primary surrogate key"),
        ("name", "character varying", "varchar", "Customer name"),
        ("tags", "ARRAY", "_varchar", "List of customer tags"),
        ("metadata", "jsonb", "jsonb", "Dynamic JSON metadata"),
    ]

    mock_conn = MockConnection(cursor_factory=lambda: MockCursor(fetchall_results=sample_columns))
    fields = SchemaFlattener.flatten_postgres_table(mock_conn, "customers", schema_name="public")

    assert len(fields) == 4

    # 1. Primitive field
    f_id = fields[0]
    assert f_id.root_column == "id"
    assert f_id.data_type == "BIGINT"
    assert f_id.is_nested is False
    assert f_id.is_array is False
    assert f_id.comment == "Primary surrogate key"

    # 2. Varchar field
    f_name = fields[1]
    assert f_name.root_column == "name"
    assert f_name.is_array is False

    # 3. Array field
    f_tags = fields[2]
    assert f_tags.root_column == "tags"
    assert f_tags.is_array is True
    assert f_tags.is_nested is True
    assert f_tags.data_type == "ARRAY<VARCHAR>"
    assert f_tags.unnest_syntax == "UNNEST(tags)"

    # 4. JSONB field
    f_meta = fields[3]
    assert f_meta.root_column == "metadata"
    assert f_meta.is_nested is True
    assert f_meta.is_array is False
    assert f_meta.data_type == "JSONB"


def test_postgres_connector_introspection():
    """Tests PostgresConnector schema introspection, PKs, FKs, and distinct value sampling."""
    table_names = [("users",), ("orders",)]
    comments = [("users", "User accounts table")]
    pks = [("users", "user_id"), ("orders", "order_id")]
    fks = [("orders", "user_id", "users", "user_id")]

    user_cols = [
        ("user_id", "integer", "int4", "User ID"),
        ("status", "character varying", "varchar", "Account status"),
    ]
    order_cols = [
        ("order_id", "integer", "int4", "Order ID"),
        ("user_id", "integer", "int4", "Foreign key to users"),
        ("order_status", "character varying", "varchar", "Status of order"),
    ]

    def cursor_dispatcher():
        cur = MagicMock()
        def mock_execute(query, params=None):
            q = str(query)
            if "information_schema.columns" in q:
                tbl = params[0] if params else ""
                cur.fetchall.return_value = user_cols if tbl == "users" else order_cols
            elif "information_schema.tables" in q:
                cur.fetchall.return_value = table_names
            elif "table_constraints" in q and "PRIMARY KEY" in q:
                cur.fetchall.return_value = pks
            elif "table_constraints" in q and "FOREIGN KEY" in q:
                cur.fetchall.return_value = fks
            elif "pg_description" in q and "pg_class" in q:
                cur.fetchall.return_value = comments
            elif "SELECT DISTINCT" in q:
                cur.fetchall.return_value = [("ACTIVE",), ("INACTIVE",)]
            else:
                cur.fetchall.return_value = []
        cur.execute = mock_execute
        cur.__enter__.return_value = cur
        return cur

    mock_conn = MagicMock()
    mock_conn.cursor = cursor_dispatcher
    mock_conn.closed = False

    connector = PostgresConnector(connection=mock_conn, schema_name="public")
    tables_meta, sample_values = connector.inspect_database_schema()

    assert "users" in tables_meta
    assert "orders" in tables_meta
    assert tables_meta["users"]["primary_keys"] == ["user_id"]
    assert tables_meta["orders"]["primary_keys"] == ["order_id"]
    assert "user_id->users.user_id" in tables_meta["orders"]["foreign_keys"]

    # Verify categorical distinct values were sampled for status columns
    assert "users" in sample_values
    assert "status" in sample_values["users"]
    assert ("ACTIVE", "ACTIVE") in sample_values["users"]["status"]


def test_postgres_connector_execute_query():
    """Tests executing queries and error handling in PostgresConnector."""
    cur = MagicMock()
    cur.description = [("id",), ("username",)]
    cur.fetchall.return_value = [(1, "alice"), (2, "bob")]
    cur.__enter__.return_value = cur

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = cur
    mock_conn.closed = False

    connector = PostgresConnector(connection=mock_conn)
    results = connector.execute_query("SELECT id, username FROM users;")
    assert len(results) == 2
    assert results[0] == {"id": 1, "username": "alice"}
    assert results[1] == {"id": 2, "username": "bob"}

    # Test error handling and rollback
    cur.execute.side_effect = Exception("syntax error at or near 'SELECTT'")
    with pytest.raises(ExecutionError) as exc_info:
        connector.execute_query("SELECTT * FROM users;")
    assert "PostgreSQL execution error" in str(exc_info.value)
    mock_conn.rollback.assert_called()


def test_text2sql_pipeline_with_postgres_backend():
    """Tests that Text2SQLPipeline initializes and runs with PostgreSQL backend."""
    cur = MagicMock()
    cur.fetchall.return_value = [("orders",)]
    cur.__enter__.return_value = cur

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = cur
    mock_conn.closed = False

    with patch.object(PostgresConnector, "inspect_database_schema") as mock_inspect, \
         patch.object(PostgresConnector, "execute_query") as mock_exec:

        mock_inspect.return_value = (
            {
                "orders": {
                    "description": "Customer order records",
                    "fields": [
                        FlattenedField(
                            table_name="orders",
                            root_column="order_id",
                            path="order_id",
                            data_type="BIGINT",
                        ),
                        FlattenedField(
                            table_name="orders",
                            root_column="customer_id",
                            path="customer_id",
                            data_type="VARCHAR",
                        ),
                        FlattenedField(
                            table_name="orders",
                            root_column="total_amount",
                            path="total_amount",
                            data_type="DOUBLE",
                        ),
                        FlattenedField(
                            table_name="orders",
                            root_column="shipping_cost",
                            path="shipping_cost",
                            data_type="DOUBLE",
                        ),
                    ],
                    "primary_keys": ["order_id"],
                    "foreign_keys": [],
                }
            },
            {},
        )
        mock_exec.return_value = [{"customer_id": "C001", "total_spent": 500.0, "shipping_ratio": 0.05}]

        pipeline = Text2SQLPipeline(
            db_backend="postgres",
            postgres_con=mock_conn,
        )

        assert pipeline.db_backend == "postgres"
        assert pipeline.pg_connector is not None

        result = pipeline.run("Find the top 3 customers ranked by total spend with safe division handling")
        assert result.get("target_dialect") == "postgres"
        assert result.get("execution_status") == "EXECUTED"
        assert result.get("execution_result") == [{"customer_id": "C001", "total_spent": 500.0, "shipping_ratio": 0.05}]
