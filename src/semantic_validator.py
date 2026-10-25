"""Semantic validator for detecting Problem 1: Erroneous Data Analysis.

Inspects SQL queries and calculation results against explicit analytical contracts:
1. Distinct entity counting vs event counting (grain mismatch).
2. Denominator distortion (dividing by events or active users instead of cohort baseline).
3. Join explosion and filter misplacement (e.g. WHERE filter turning LEFT JOIN into INNER JOIN).
4. Safe math guards (NULLIF on division).
5. Mathematical bounds verification (e.g., retention rate cannot exceed 100.0%).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)


class AnalyticalContract(BaseModel):
    """Explicit business and analytical specification for a query."""

    metric_name: str = Field(description="Name of the target metric (e.g., 'User Retention Rate')")
    target_entity: str = Field(default="user_id", description="Grain entity to count (e.g. 'user_id', 'loan_ref')")
    requires_distinct_entity_count: bool = Field(
        default=True, description="True if the calculation must count distinct entities rather than events"
    )
    is_percentage: bool = Field(default=True, description="True if the metric is a percentage (0% - 100%)")
    baseline_cohort_filter: Optional[str] = Field(
        default=None, description="Description of the starting cohort (denominator)"
    )
    returning_cohort_filter: Optional[str] = Field(
        default=None, description="Description of the returning activity condition (numerator)"
    )


class SemanticValidationReport(BaseModel):
    """Detailed diagnosis of analytical and semantic pitfalls."""

    is_semantically_sound: bool = Field(description="True if query matches intended business logic without pitfalls")
    warnings: List[str] = Field(default_factory=list, description="List of suspicious logic patterns detected")
    errors: List[str] = Field(default_factory=list, description="Definite logic errors that invalidate the answer")
    detected_grain: str = Field(default="unknown", description="Detected counting grain in SQL")
    has_distinct_entity_count: bool = Field(default=False, description="True if SQL uses COUNT(DISTINCT entity)")
    has_safe_division: bool = Field(default=False, description="True if division is guarded by NULLIF")
    has_join_filter_leak: bool = Field(
        default=False, description="True if WHERE clause filters out non-returning cohort members on LEFT JOIN"
    )
    result_bounds_valid: bool = Field(default=True, description="True if returned numeric result is within plausible range")


def validate_semantic_analysis(
    sql: str,
    contract: AnalyticalContract,
    actual_result: Optional[float] = None,
    dialect: str = "postgres",
) -> SemanticValidationReport:
    """Analyzes SQL AST and execution results for semantic correctness against an AnalyticalContract."""
    clean_sql = sql.strip().rstrip(";")
    warnings: List[str] = []
    errors: List[str] = []

    try:
        ast = sqlglot.parse_one(clean_sql, read=dialect)
    except Exception as exc:
        return SemanticValidationReport(
            is_semantically_sound=False,
            errors=[f"Failed to parse SQL for semantic validation: {exc}"],
        )

    # 1. Inspect Entity Counting Grain
    # Find all COUNT expressions
    count_funcs = [f for f in ast.find_all(exp.Count)]
    has_distinct_target_entity = False
    counts_non_distinct = False

    for c in count_funcs:
        # Check if COUNT has DISTINCT
        is_distinct = bool(c.find(exp.Distinct))
        # Inspect arguments
        arg_str = str(c.this).lower() if c.this else ""
        if contract.target_entity.lower() in arg_str:
            if is_distinct:
                has_distinct_target_entity = True
            else:
                counts_non_distinct = True
        elif arg_str in ("*", "1") or "event_id" in arg_str:
            counts_non_distinct = True

    if contract.requires_distinct_entity_count and not has_distinct_target_entity:
        errors.append(
            f"Grain Mismatch: Analytical contract requires distinct counting of '{contract.target_entity}', "
            f"but query counts rows/events or non-distinct keys. This leads to duplicate counts from repeat activity."
        )

    # 2. Inspect Join Filter Misplacement (The LEFT JOIN -> INNER JOIN trap)
    # If there is a LEFT JOIN to an activity table, but the WHERE clause has conditions on that table,
    # non-active cohort members are filtered out, corrupting the denominator!
    has_join_filter_leak = False
    left_joins = [j for j in ast.find_all(exp.Join) if (j.side or "").upper() == "LEFT"]
    where_clause = ast.find(exp.Where)

    if left_joins and where_clause:
        joined_table_aliases = set()
        for j in left_joins:
            if j.this:
                alias = j.this.alias_or_name
                if alias:
                    joined_table_aliases.add(alias.lower())

        where_cols = {c.table.lower() for c in where_clause.find_all(exp.Column) if c.table}
        leaked = joined_table_aliases.intersection(where_cols)
        if leaked:
            has_join_filter_leak = True
            errors.append(
                f"Filter Misplacement Trap: LEFT JOIN on table alias(es) {list(leaked)} has filtering predicates "
                f"in the WHERE clause. This converts the LEFT JOIN into an INNER JOIN, discarding users who did not return "
                f"and corrupting the cohort denominator."
            )

    # 3. Inspect Safe Division Guard
    divisions = list(ast.find_all(exp.Div))
    has_safe_division = True
    if divisions:
        for div in divisions:
            # Check right-hand side of division for NULLIF
            rhs = div.right
            if rhs and not rhs.find(exp.Nullif):
                has_safe_division = False
                warnings.append(
                    "Division without NULLIF guard: Ratios should wrap the denominator in NULLIF(denominator, 0) "
                    "to prevent division-by-zero crashes on empty cohorts."
                )

    # 4. Result Bounds Checking (e.g. Percentage cannot be > 100% or < 0%)
    result_bounds_valid = True
    if actual_result is not None and contract.is_percentage:
        if actual_result < 0.0 or actual_result > 100.0:
            result_bounds_valid = False
            errors.append(
                f"Plausibility Violation: Computed percentage is {actual_result:.2f}%, which is outside "
                f"the valid range of [0.0%, 100.0%]. This proves a grain mismatch or inverted denominator."
            )

    is_sound = (len(errors) == 0) and result_bounds_valid

    detected_grain = "distinct_" + contract.target_entity if has_distinct_target_entity else "row/event_level"

    return SemanticValidationReport(
        is_semantically_sound=is_sound,
        warnings=warnings,
        errors=errors,
        detected_grain=detected_grain,
        has_distinct_entity_count=has_distinct_target_entity,
        has_safe_division=has_safe_division,
        has_join_filter_leak=has_join_filter_leak,
        result_bounds_valid=result_bounds_valid,
    )
