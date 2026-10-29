"""Unit tests for Problem 2: Schema Linking & Large-Scale Schema Pruning."""

from __future__ import annotations

import pytest
from src.query_executor import MCPQueryExecutor
from src.schema_graph import SchemaGraph
from src.schema_pruner import SchemaPruner


@pytest.fixture
def executor():
    return MCPQueryExecutor()


@pytest.fixture
def pruner():
    return SchemaPruner(target="postgres")


@pytest.fixture
def graph():
    return SchemaGraph(target="postgres")


def test_schema_graph_shortest_path(graph):
    """Verifies BFS path discovery across single-hop and multi-hop relational joins."""
    # 1. Single hop: staff -> outlets
    p1 = graph.find_shortest_join_path("staff", "outlets")
    assert p1 is not None
    assert len(p1) == 1
    assert p1[0].join_predicate == "staff.outlet_code = outlets.outlet_code"

    # 2. Multi-hop: staff_attendance -> outlets (via staff)
    p2 = graph.find_shortest_join_path("staff_attendance", "outlets")
    assert p2 is not None
    assert len(p2) == 2
    predicates = [step.join_predicate for step in p2]
    assert "staff_attendance.staff_no = staff.staff_no" in predicates
    assert "staff.outlet_code = outlets.outlet_code" in predicates


def test_coarse_table_pruning(pruner):
    """Verifies that relevant candidate tables are retrieved and non-relevant tables are pruned."""
    # 1. Attendance query
    r1 = pruner.prune("What was the staff attendance in June?")
    assert "staff_attendance" in r1.candidate_tables
    assert "staff" in r1.candidate_tables
    assert "group_loans" not in r1.candidate_tables

    # 2. Retail loans query
    r2 = pruner.prune("What is the rejection rate for individual loans by product?")
    assert "individual_loans" in r2.candidate_tables
    assert "staff_attendance" not in r2.candidate_tables


def test_fine_column_pruning_and_token_reduction(pruner):
    """Verifies that irrelevant columns are pruned while preserving PK/FKs and core metrics."""
    q = "What was the rejection rate for individual loans by product?"
    report = pruner.prune(q)

    # In individual_loans (25 columns), audit/internal columns must be pruned
    indiv_schema = report.pruned_schemas.get("individual_loans")
    assert indiv_schema is not None
    assert indiv_schema.pruned_column_count < indiv_schema.original_column_count
    assert report.token_reduction_pct > 50.0

    selected_col_names = {c["name"] for c in indiv_schema.selected_columns}
    assert "app_no" in selected_col_names  # Primary Key
    assert "loan_product" in selected_col_names  # Mentioned
    assert "outlet_head_id" in indiv_schema.dropped_columns  # Irrelevant audit column dropped


def test_mcp_prune_schema_tool(executor):
    """Verifies MCP tool 'prune_schema' executes cleanly via MCP client."""
    res = executor.prune_schema_sync("Show staff attendance for casual leave in June", target="postgres")

    assert "candidate_tables" in res
    assert "compact_schema_markdown" in res
    assert "token_reduction_pct" in res
    assert len(res["candidate_tables"]) >= 1


def test_mcp_find_join_path_tool(executor):
    """Verifies MCP tool 'find_join_path' returns shortest join paths."""
    res = executor.find_join_path_sync("staff_attendance", "outlets", target="postgres")

    assert res["found_path"] is True
    assert res["path_length"] == 2
    assert len(res["join_predicates"]) == 2


def test_mcp_get_schema_graph_tool(executor):
    """Verifies MCP tool 'get_schema_graph' returns tables and relationships."""
    res = executor.get_schema_graph_sync(target="postgres")

    assert "tables" in res
    assert "relationships" in res
    assert "outlets" in res["tables"]
    assert len(res["relationships"]) >= 5
