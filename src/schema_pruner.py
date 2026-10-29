"""Schema Pruner module (Problem 2: Schema Linking & Large-Scale Schema Pruning).

Implements:
1. Coarse Table Pruning (Entity & Table Retrieval / Relevance Scoring).
2. Fine Column Pruning (Filtering irrelevant columns, preserving PK/FK join paths).
3. Compact Schema Card Generation with Token Reduction metrics.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.literal_resolver import LiteralResolver
from src.schema_graph import LENDING_TABLES_METADATA, SchemaGraph
from src.schema_inspector import SchemaInspector

logger = logging.getLogger(__name__)


class TablePruningScore(BaseModel):
    """Relevance score and match rationale for a database table."""

    table_name: str
    score: float
    relevance_reasons: List[str]
    is_selected: bool


class PrunedTableSchema(BaseModel):
    """Pruned table representation with only candidate columns."""

    table_name: str
    relevance_score: float
    original_column_count: int
    pruned_column_count: int
    selected_columns: List[Dict[str, Any]]
    dropped_columns: List[str]
    join_keys: List[str]


class SchemaPruningReport(BaseModel):
    """Complete Problem 2 schema linking and pruning report."""

    user_query: str
    candidate_tables: List[str]
    table_scores: List[TablePruningScore]
    pruned_schemas: Dict[str, PrunedTableSchema]
    discovered_join_paths: List[str]
    original_total_columns: int
    pruned_total_columns: int
    token_reduction_pct: float
    compact_schema_markdown: str


class SchemaPruner:
    """Ranks, selects, and prunes tables and columns to prevent context bloat and hallucinated joins."""

    def __init__(self, target: str = "postgres"):
        self.target = target.lower()
        self.inspector = SchemaInspector(target=self.target)
        self.graph = SchemaGraph(target=self.target)
        self.resolver = LiteralResolver(target=self.target)

    def prune(self, user_query: str, top_k_tables: int = 3) -> SchemaPruningReport:
        """Executes full Coarse Table Pruning and Fine Column Pruning for a query."""
        q_lower = user_query.lower()
        all_tables = self.inspector.get_table_names()

        # Step 1: Discover Grounded Literals via MCP
        grounded_literals = self.resolver.resolve_literals(user_query)
        grounded_tables = {m.table_name for m in grounded_literals}

        # Step 2: Coarse Table Pruning (Relevance Scoring)
        table_scores: List[TablePruningScore] = []
        for table in all_tables:
            score = 0.0
            reasons: List[str] = []

            # 2a. Grounded literal match (+40)
            if table in grounded_tables:
                score += 40.0
                matched_terms = [m.term for m in grounded_literals if m.table_name == table]
                reasons.append(f"Grounded literal match: {', '.join(matched_terms)}")

            # 2b. Table name direct mention (+30)
            table_words = table.replace("_", " ").split()
            for w in table_words:
                w_stem = w.rstrip("s")
                if len(w_stem) >= 3 and re.search(r"\b" + re.escape(w_stem), q_lower):
                    score += 30.0
                    reasons.append(f"Table name match '{w}'")
                    break

            # 2c. Semantic domain tag match (+20 each)
            meta = LENDING_TABLES_METADATA.get(table)
            if meta:
                for tag in meta.domain_tags:
                    tag_stem = tag.rstrip("s")
                    if re.search(r"\b" + re.escape(tag) + r"\b", q_lower) or (len(tag_stem) >= 4 and re.search(r"\b" + re.escape(tag_stem), q_lower)):
                        score += 20.0
                        reasons.append(f"Domain keyword match '{tag}'")

            # 2d. Column name match (+15 each)
            try:
                cols = self.inspector.get_table_columns(table)
                for c in cols:
                    c_clean = c["name"].replace("_", " ").lower()
                    c_stem = c_clean.rstrip("s")
                    if len(c_stem) >= 4 and re.search(r"\b" + re.escape(c_stem), q_lower):
                        score += 15.0
                        reasons.append(f"Column match '{c['name']}'")
            except Exception:
                pass

            table_scores.append(TablePruningScore(
                table_name=table,
                score=score,
                relevance_reasons=reasons,
                is_selected=False,
            ))

        # Sort tables by descending score
        table_scores.sort(key=lambda x: x.score, reverse=True)

        # Select candidate tables: Top-K with score > 0 (fallback to top 2 if no keywords)
        selected_table_names: List[str] = []
        for item in table_scores:
            if item.score > 0 and len(selected_table_names) < top_k_tables:
                item.is_selected = True
                selected_table_names.append(item.table_name)

        if not selected_table_names:
            # Fallback default: outlets and staff
            selected_table_names = all_tables[:2]
            for item in table_scores:
                if item.table_name in selected_table_names:
                    item.is_selected = True

        # Ensure join continuity: If two candidate tables need an intermediate bridge table, add it!
        if len(selected_table_names) >= 2:
            path = self.graph.find_shortest_join_path(selected_table_names[0], selected_table_names[1])
            if path:
                for edge in path:
                    for tbl in (edge.source_table, edge.target_table):
                        if tbl not in selected_table_names and len(selected_table_names) < top_k_tables + 1:
                            selected_table_names.append(tbl)
                            for ts in table_scores:
                                if ts.table_name == tbl:
                                    ts.is_selected = True
                                    ts.relevance_reasons.append("Bridge table for join path continuity")

        # Step 3: Discover Active Join Paths between candidate tables
        discovered_paths: List[str] = []
        seen_pairs: Set[Tuple[str, str]] = set()
        for i in range(len(selected_table_names)):
            for j in range(i + 1, len(selected_table_names)):
                p = self.graph.find_shortest_join_path(selected_table_names[i], selected_table_names[j])
                if p:
                    for edge in p:
                        pair = tuple(sorted([f"{edge.source_table}.{edge.source_column}", f"{edge.target_table}.{edge.target_column}"]))
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            discovered_paths.append(edge.join_predicate)

        # Step 4: Fine Column Pruning
        pruned_schemas: Dict[str, PrunedTableSchema] = {}
        total_orig_cols = 0
        total_pruned_cols = 0

        # Collect columns essential for join predicates
        essential_join_columns: Set[Tuple[str, str]] = set()
        for edge in self.graph.edges:
            if edge.source_table in selected_table_names and edge.target_table in selected_table_names:
                essential_join_columns.add((edge.source_table, edge.source_column))
                essential_join_columns.add((edge.target_table, edge.target_column))

        for tbl in selected_table_names:
            cols = self.inspector.get_table_columns(tbl)
            total_orig_cols += len(cols)

            selected_cols: List[Dict[str, Any]] = []
            dropped_cols: List[str] = []
            tbl_score_val = next((ts.score for ts in table_scores if ts.table_name == tbl), 0.0)

            # Retrieve table metadata from logical graph if available
            table_meta = LENDING_TABLES_METADATA.get(tbl)
            logical_pks = set(table_meta.primary_key) if table_meta else set()
            important_metrics = set(table_meta.important_metrics) if table_meta else set()

            for c in cols:
                c_name = c["name"]
                c_clean = c_name.replace("_", " ").lower()

                # Rule 1: Always keep Primary Keys (physical or logical schema graph)
                is_pk = c.get("is_primary_key", False) or (c_name in logical_pks)
                # Rule 2: Always keep Foreign Keys used in active joins
                is_fk = (tbl, c_name) in essential_join_columns
                # Rule 3: Keep if column is mentioned in query
                is_mentioned = c_clean in q_lower or c_name.lower() in q_lower
                # Rule 4: Keep if column matches a grounded literal
                is_literal_col = any(m.column_name == c_name and m.table_name == tbl for m in grounded_literals)
                # Rule 5: Keep core domain attributes & important metrics
                is_core = (
                    c_name in {"state_name", "tier", "cluster_name", "position", "stage", "attendance_code", "attendance_date", "loan_product", "paid_amt", "payout_amt"}
                    or c_name in important_metrics
                )

                if is_pk or is_fk or is_mentioned or is_literal_col or is_core:
                    selected_cols.append(c)
                else:
                    dropped_cols.append(c_name)

            # Ensure at least 3 columns are kept if table was selected
            if len(selected_cols) < 3 and cols:
                for c in cols[:4]:
                    if c not in selected_cols:
                        selected_cols.append(c)
                        if c["name"] in dropped_cols:
                            dropped_cols.remove(c["name"])

            total_pruned_cols += len(selected_cols)
            pruned_schemas[tbl] = PrunedTableSchema(
                table_name=tbl,
                relevance_score=tbl_score_val,
                original_column_count=len(cols),
                pruned_column_count=len(selected_cols),
                selected_columns=selected_cols,
                dropped_columns=dropped_cols,
                join_keys=[c["name"] for c in selected_cols if (tbl, c["name"]) in essential_join_columns],
            )

        # Step 5: Format Compact Markdown Schema Cards
        card_lines = ["## Compact Pruned Schema Context (Problem 2)", ""]
        if discovered_paths:
            card_lines.append("### Active Relational Join Paths:")
            for p in discovered_paths:
                card_lines.append(f"- `JOIN ... ON {p}`")
            card_lines.append("")

        for tbl, ps in pruned_schemas.items():
            card_lines.append(f"### Table: `{tbl}` (Relevance: {ps.relevance_score:.0f} pts | Retained {ps.pruned_column_count}/{ps.original_column_count} columns)")
            card_lines.append("| Column | Type | Role |")
            card_lines.append("|---|---|---|")
            for c in ps.selected_columns:
                role = "PRIMARY KEY" if c.get("is_primary_key") else ("JOIN KEY" if c["name"] in ps.join_keys else "-")
                card_lines.append(f"| `{c['name']}` | `{c['data_type']}` | {role} |")
            if ps.dropped_columns:
                card_lines.append(f"*Pruned {len(ps.dropped_columns)} irrelevant columns: {', '.join(ps.dropped_columns[:6])}{'...' if len(ps.dropped_columns) > 6 else ''}*")
            card_lines.append("")

        reduction = ((total_orig_cols - total_pruned_cols) / total_orig_cols * 100.0) if total_orig_cols else 0.0

        return SchemaPruningReport(
            user_query=user_query,
            candidate_tables=selected_table_names,
            table_scores=table_scores,
            pruned_schemas=pruned_schemas,
            discovered_join_paths=discovered_paths,
            original_total_columns=total_orig_cols,
            pruned_total_columns=total_pruned_cols,
            token_reduction_pct=round(reduction, 1),
            compact_schema_markdown="\n".join(card_lines).strip(),
        )
