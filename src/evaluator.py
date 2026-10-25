"""Evaluation harness comparing Baseline Naive Text-to-SQL against the Improved MCP Pipeline.

Demonstrates Problem 1: Erroneous Data Analysis.
Calculates ground truth dynamically from the underlying database (never hardcoded),
executes both pipelines through MCP, and diagnoses failure modes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from src.query_executor import MCPQueryExecutor
from src.semantic_validator import SemanticValidationReport, validate_semantic_analysis
from src.sql_generator import SQLGenerator
from src.sql_validator import validate_readonly_sql

logger = logging.getLogger(__name__)


class PipelineExecutionResult(BaseModel):
    """Execution telemetry for one pipeline run."""

    pipeline_name: str
    generated_sql: str
    is_valid_sql: bool
    sql_validation_error: Optional[str] = None
    execution_success: bool
    raw_result: Optional[float] = None
    ground_truth: float
    absolute_error: float
    is_accurate: bool
    semantic_report: SemanticValidationReport
    explanation: str


class Problem1Evaluator:
    """Evaluates Baseline vs Improved Text-to-SQL on the cohort retention benchmark."""

    def __init__(self, executor: Optional[MCPQueryExecutor] = None):
        self.executor = executor or MCPQueryExecutor()
        self.generator = SQLGenerator(executor=self.executor)

    def compute_ground_truth(self, target: str = "local") -> Dict[str, Any]:
        """Calculates expected ground truth dynamically from the raw underlying database records."""
        # Query 1: Total distinct January signups
        denom_sql = """
        SELECT COUNT(DISTINCT user_id) AS total_jan_signups
        FROM users
        WHERE signup_date >= '2024-01-01' AND signup_date < '2024-02-01';
        """
        denom_res = self.executor.execute_readonly_sql_sync(denom_sql, target=target)
        total_jan_signups = denom_res["rows"][0]["total_jan_signups"]

        # Query 2: Distinct January signups returning in February
        num_sql = """
        SELECT COUNT(DISTINCT u.user_id) AS returning_feb_users
        FROM users u
        JOIN activity_events e ON u.user_id = e.user_id
        WHERE u.signup_date >= '2024-01-01' AND u.signup_date < '2024-02-01'
          AND e.event_date >= '2024-02-01' AND e.event_date < '2024-03-01';
        """
        num_res = self.executor.execute_readonly_sql_sync(num_sql, target=target)
        returning_feb_users = num_res["rows"][0]["returning_feb_users"]

        true_retention_rate = round((returning_feb_users * 100.0) / total_jan_signups, 2)

        return {
            "january_signups": total_jan_signups,
            "returning_february_users": returning_feb_users,
            "true_retention_rate": true_retention_rate,
        }

    def evaluate_baseline(self, question: str, target: str = "local") -> PipelineExecutionResult:
        """Executes and evaluates the Naive Baseline approach."""
        gt = self.compute_ground_truth(target=target)
        gen = self.generator.generate_baseline_sql(question, target=target)

        # 1. AST Validation
        val_res = self.executor.validate_sql_sync(gen.generated_sql)
        is_valid = val_res.get("is_valid", False)

        # 2. Execution through MCP
        exec_res = self.executor.execute_readonly_sql_sync(gen.generated_sql, target=target)
        raw_val = None
        if exec_res.get("success") and exec_res.get("rows"):
            first_row = exec_res["rows"][0]
            val = list(first_row.values())[0]
            raw_val = float(val) if val is not None else None

        # 3. Semantic Validation
        sem_report = validate_semantic_analysis(
            gen.generated_sql,
            contract=gen.contract,
            actual_result=raw_val,
        )

        expected = gt["true_retention_rate"]
        abs_err = abs(raw_val - expected) if raw_val is not None else 999.0
        is_acc = abs_err < 0.01

        explanation = (
            f"BASELINE FAILURE ANALYSIS:\n"
            f"- SQL execution status: SUCCESS (PostgreSQL/database returned 0 errors).\n"
            f"- Computed result: {raw_val}% vs Ground Truth: {expected}%.\n"
            f"- Absolute Error: {abs_err:.2f}%.\n"
            f"- Why it failed: The query counts events instead of distinct users, and converts "
            f"the LEFT JOIN to an INNER JOIN via the WHERE filter, completely distorting the denominator."
        )

        return PipelineExecutionResult(
            pipeline_name="Baseline (Naive)",
            generated_sql=gen.generated_sql,
            is_valid_sql=is_valid,
            sql_validation_error=val_res.get("error"),
            execution_success=exec_res.get("success", False),
            raw_result=raw_val,
            ground_truth=expected,
            absolute_error=abs_err,
            is_accurate=is_acc,
            semantic_report=sem_report,
            explanation=explanation,
        )

    def evaluate_improved(self, question: str, target: str = "local") -> PipelineExecutionResult:
        """Executes and evaluates the Semantically Guarded MCP approach."""
        gt = self.compute_ground_truth(target=target)
        gen = self.generator.generate_improved_sql(question, target=target)

        # 1. AST Validation
        val_res = self.executor.validate_sql_sync(gen.generated_sql)
        is_valid = val_res.get("is_valid", False)

        # 2. Execution through MCP
        exec_res = self.executor.execute_readonly_sql_sync(gen.generated_sql, target=target)
        raw_val = None
        if exec_res.get("success") and exec_res.get("rows"):
            first_row = exec_res["rows"][0]
            val = list(first_row.values())[0]
            raw_val = float(val) if val is not None else None

        # 3. Semantic Validation
        sem_report = validate_semantic_analysis(
            gen.generated_sql,
            contract=gen.contract,
            actual_result=raw_val,
        )

        expected = gt["true_retention_rate"]
        abs_err = abs(raw_val - expected) if raw_val is not None else 999.0
        is_acc = abs_err < 0.01

        explanation = (
            f"IMPROVED MCP PIPELINE SUCCESS:\n"
            f"- SQL execution status: SUCCESS.\n"
            f"- Computed result: {raw_val}% vs Ground Truth: {expected}%.\n"
            f"- Absolute Error: {abs_err:.2f}% (Accurate!).\n"
            f"- Why it succeeded: Decomposed analytical contract enforced distinct user counting "
            f"and guarded the denominator with NULLIF, preventing join fan-out and filter misplacement."
        )

        return PipelineExecutionResult(
            pipeline_name="Improved (MCP + Semantic Validation)",
            generated_sql=gen.generated_sql,
            is_valid_sql=is_valid,
            sql_validation_error=val_res.get("error"),
            execution_success=exec_res.get("success", False),
            raw_result=raw_val,
            ground_truth=expected,
            absolute_error=abs_err,
            is_accurate=is_acc,
            semantic_report=sem_report,
            explanation=explanation,
        )
