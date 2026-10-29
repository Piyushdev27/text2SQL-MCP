"""Schema Knowledge Graph module (Problem 2: Schema Linking).

Defines the relational graph, logical foreign key relationships,
and shortest join path resolution for enterprise Text-to-SQL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class JoinEdge(BaseModel):
    """Represents a logical or physical relational join between two tables."""

    source_table: str = Field(description="Originating table")
    source_column: str = Field(description="Column on originating table")
    target_table: str = Field(description="Referenced table")
    target_column: str = Field(description="Column on referenced table")
    relationship: str = Field(default="many-to-one", description="'one-to-many', 'many-to-one', or 'one-to-one'")
    join_predicate: str = Field(description="Ready-to-use ON clause, e.g. staff.outlet_code = outlets.outlet_code")


class TableMetadata(BaseModel):
    """Semantic metadata for a database table."""

    name: str
    description: str
    primary_key: List[str]
    domain_tags: List[str]
    important_metrics: List[str] = Field(default_factory=list)


# Logical Knowledge Graph for lending_sample PostgreSQL warehouse
LENDING_TABLES_METADATA: Dict[str, TableMetadata] = {
    "outlets": TableMetadata(
        name="outlets",
        description="Physical branch outlets across Indian states, clusters, zones, and operational tiers (T1, T2, T3).",
        primary_key=["outlet_code"],
        domain_tags=["branches", "location", "geography", "states", "tiers", "clusters", "zones"],
        important_metrics=["opened_on", "tier", "state_name", "cluster_name"],
    ),
    "staff": TableMetadata(
        name="staff",
        description="Employees and field officers, their assigned branch outlet, and employment status.",
        primary_key=["staff_no"],
        domain_tags=["employees", "personnel", "workers", "officers", "positions", "managers"],
        important_metrics=["position", "joined_on", "staff_status"],
    ),
    "staff_attendance": TableMetadata(
        name="staff_attendance",
        description="Daily attendance logs recording presence (PR), casual leave (CL), sick leave (SL), week off (WO), absent (AB).",
        primary_key=["staff_no", "attendance_date"],
        domain_tags=["attendance", "present", "absent", "leave", "shifts", "workdays"],
        important_metrics=["attendance_code", "attendance_date"],
    ),
    "group_loans": TableMetadata(
        name="group_loans",
        description="Joint-liability group loans, disbursements, requested amounts, schemes, and application stages.",
        primary_key=["loan_ref"],
        domain_tags=["group loans", "microfinance", "disbursements", "payouts", "schemes", "jlg"],
        important_metrics=["requested_amt", "approved_amt", "paid_amt", "paid_at", "stage"],
    ),
    "individual_loans": TableMetadata(
        name="individual_loans",
        description="Individual retail loan applications, underwriting, product lines, rejection status, and payouts.",
        primary_key=["app_no"],
        domain_tags=["individual loans", "retail", "products", "underwriting", "rejections", "applications"],
        important_metrics=["loan_product", "file_status", "sanctioned_amt", "payout_amt", "payout_at", "rejected_at"],
    ),
    "outlet_targets": TableMetadata(
        name="outlet_targets",
        description="Monthly disbursement targets for group and individual portfolios assigned to branch outlets.",
        primary_key=["outlet_code", "target_day"],
        domain_tags=["targets", "budgets", "goals", "quotas", "kpi"],
        important_metrics=["group_target", "indiv_target"],
    ),
    "overdue_daily": TableMetadata(
        name="overdue_daily",
        description="Daily delinquency tracking, portfolio-at-risk aging buckets (1-30, 31-60, 61-90, 90+ days), and dues.",
        primary_key=["outlet_code", "as_of_date", "loan_type"],
        domain_tags=["overdue", "delinquency", "npa", "par", "dues", "collections"],
        important_metrics=["dues_today", "overdue_accounts", "opening_90_plus"],
    ),
}

# Logical Join Relationships (PK-FK graph)
LENDING_RELATIONAL_EDGES: List[JoinEdge] = [
    JoinEdge(
        source_table="staff",
        source_column="outlet_code",
        target_table="outlets",
        target_column="outlet_code",
        relationship="many-to-one",
        join_predicate="staff.outlet_code = outlets.outlet_code",
    ),
    JoinEdge(
        source_table="staff_attendance",
        source_column="staff_no",
        target_table="staff",
        target_column="staff_no",
        relationship="many-to-one",
        join_predicate="staff_attendance.staff_no = staff.staff_no",
    ),
    JoinEdge(
        source_table="group_loans",
        source_column="outlet_code",
        target_table="outlets",
        target_column="outlet_code",
        relationship="many-to-one",
        join_predicate="group_loans.outlet_code = outlets.outlet_code",
    ),
    JoinEdge(
        source_table="individual_loans",
        source_column="outlet_id",
        target_table="outlets",
        target_column="outlet_code",
        relationship="many-to-one",
        join_predicate="individual_loans.outlet_id = outlets.outlet_code",
    ),
    JoinEdge(
        source_table="outlet_targets",
        source_column="outlet_code",
        target_table="outlets",
        target_column="outlet_code",
        relationship="many-to-one",
        join_predicate="outlet_targets.outlet_code = outlets.outlet_code",
    ),
    JoinEdge(
        source_table="overdue_daily",
        source_column="outlet_code",
        target_table="outlets",
        target_column="outlet_code",
        relationship="many-to-one",
        join_predicate="overdue_daily.outlet_code = outlets.outlet_code",
    ),
]


class SchemaGraph:
    """Graph manager for relational database schema linking and join path discovery."""

    def __init__(self, target: str = "postgres"):
        self.target = target.lower()
        self.tables = LENDING_TABLES_METADATA
        self.edges = LENDING_RELATIONAL_EDGES
        self._adjacency: Dict[str, List[Tuple[str, JoinEdge]]] = {}
        self._build_adjacency()

    def _build_adjacency(self) -> None:
        """Constructs bidirectional graph adjacency representation."""
        for t in self.tables:
            self._adjacency[t] = []

        for edge in self.edges:
            # Forward edge
            self._adjacency[edge.source_table].append((edge.target_table, edge))
            # Reverse edge (invert predicate if traversed backwards)
            rev_predicate = f"{edge.target_table}.{edge.target_column} = {edge.source_table}.{edge.source_column}"
            rev_edge = JoinEdge(
                source_table=edge.target_table,
                source_column=edge.target_column,
                target_table=edge.source_table,
                target_column=edge.source_column,
                relationship="one-to-many" if edge.relationship == "many-to-one" else "many-to-one",
                join_predicate=rev_predicate,
            )
            self._adjacency[edge.target_table].append((edge.source_table, rev_edge))

    def find_shortest_join_path(self, start_table: str, end_table: str) -> Optional[List[JoinEdge]]:
        """Finds shortest relational path between two tables using Breadth-First Search (BFS)."""
        if start_table == end_table:
            return []

        if start_table not in self._adjacency or end_table not in self._adjacency:
            return None

        queue: List[Tuple[str, List[JoinEdge]]] = [(start_table, [])]
        visited: Set[str] = {start_table}

        while queue:
            current, path = queue.pop(0)

            if current == end_table:
                return path

            for neighbor, edge in self._adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [edge]))

        return None

    def get_full_graph_definition(self) -> Dict[str, Any]:
        """Returns serialized schema graph representation for MCP tools and resources."""
        return {
            "database": self.target,
            "tables": {name: meta.model_dump() for name, meta in self.tables.items()},
            "relationships": [e.model_dump() for e in self.edges],
        }
