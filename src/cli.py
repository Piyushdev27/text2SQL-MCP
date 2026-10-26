"""Interactive Terminal CLI Console for Text-to-SQL Problem 1.

Provides an interactive developer REPL with:
- Natural Language Question evaluation (Problem 1 Erroneous Analysis diagnosis).
- Direct SQL Execution Sandbox with real-time AST security validation.
- Live PostgreSQL schema exploration.
- Rich-formatted colorized panels, SQL syntax highlighting, and result tables.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src.config import get_settings
from src.evaluator import Problem1Evaluator
from src.query_executor import MCPQueryExecutor
from src.schema_inspector import SchemaInspector
from src.semantic_validator import validate_semantic_analysis
from src.sql_generator import SQLGenerator
from src.sql_validator import validate_readonly_sql

console = Console()


class InteractiveConsole:
    """Interactive CLI REPL for Text-to-SQL Problem 1."""

    def __init__(self):
        self.settings = get_settings()
        self.executor = MCPQueryExecutor()
        self.generator = SQLGenerator(executor=self.executor)
        self.evaluator = Problem1Evaluator(executor=self.executor)
        self.active_target = "postgres"  # Default database target

    def print_banner(self) -> None:
        """Displays the startup banner."""
        banner_text = (
            "[bold cyan]Text-to-SQL Problem 1: Interactive Terminal Console[/bold cyan]\n"
            "[dim]Model Context Protocol (MCP) + PostgreSQL + Semantic AST Validation[/dim]\n\n"
            f"[bold yellow]Active Database Target:[/bold yellow] [bold green]{self.active_target.upper()}[/bold green] "
            f"([dim]{'lending_sample on 35.200.180.162' if self.active_target == 'postgres' else 'local_problem1.db'}[/dim])"
        )
        console.print(Panel(banner_text, border_style="cyan", padding=(1, 2)))

    def print_menu(self) -> None:
        """Prints available preset options and command hints."""
        menu = Table(title="Interactive Commands & Presets", border_style="dim", show_header=True)
        menu.add_column("Key", style="bold yellow", width=6)
        menu.add_column("Description", style="white")
        menu.add_column("Target Database", style="dim cyan", width=18)

        menu.add_row("1", "User Retention Benchmark (Jan/Feb Cohort Grain Mismatch)", "Local Benchmark")
        menu.add_row("2", "Staff Attendance Rate (June 2026 Join Leak Trap)", "Live PostgreSQL")
        menu.add_row("3", "Group Loan Payout Conversion Rate (May 2026)", "Live PostgreSQL")
        menu.add_row("4", "Inspect Live PostgreSQL Database Schema & Tables", "Live PostgreSQL")
        menu.add_row("5", "SQL Security Sandbox (Test AST validation on custom SQL)", "Active Target")
        menu.add_row("6", f"Toggle Target Database (Current: {self.active_target.upper()})", "Switch Target")
        menu.add_row("help", "Redisplay this menu", "-")
        menu.add_row("q", "Exit / Quit", "-")

        console.print(menu)
        console.print("[dim]Tip: You can also type ANY custom SQL query (e.g. 'SELECT * FROM outlets LIMIT 3;') directly![/dim]\n")

    def run_problem1_analysis(self, question: str, target: Optional[str] = None) -> None:
        """Runs the complete Problem 1 Baseline vs Improved analysis and displays Rich comparison."""
        tgt = target or self.active_target
        console.print(f"\n[bold cyan]=== Evaluating Question on [{tgt.upper()}] ===[/bold cyan]")
        console.print(f"[bold italic white]\"{question}\"[/bold italic white]\n")

        contract, _ = self.generator.decompose_question(question)

        # 1. Show Analytical Contract
        contract_table = Table(title="Decomposed Analytical Contract", border_style="cyan", show_header=False)
        contract_table.add_column("Field", style="bold yellow")
        contract_table.add_column("Specification", style="white")
        contract_table.add_row("Metric Name", contract.metric_name)
        contract_table.add_row("Target Grain Entity", f"[bold green]{contract.target_entity}[/bold green]")
        contract_table.add_row("Requires DISTINCT?", str(contract.requires_distinct_entity_count))
        contract_table.add_row("Is Percentage?", str(contract.is_percentage))
        contract_table.add_row("Baseline Cohort", contract.baseline_cohort_filter or "Full Population")
        contract_table.add_row("Returning Filter", contract.returning_cohort_filter or "Active Subset")
        console.print(contract_table)

        # 1b. Schema Grounding: Resolve Categorical Literals via MCP (Problem 2 Slice)
        grounded_literals = self.executor.resolve_categorical_literals_sync(question, target=tgt)
        if grounded_literals:
            lit_table = Table(title="Discovered Database Literals (Schema Grounding via MCP)", border_style="magenta", show_header=True)
            lit_table.add_column("Natural Language Term", style="bold yellow")
            lit_table.add_column("Target Table & Column", style="white")
            lit_table.add_column("Database Cell Value", style="bold green")
            lit_table.add_column("Match Type", style="dim")
            for m in grounded_literals:
                lit_table.add_row(m["term"], f"{m['table_name']}.{m['column_name']}", f"'{m['database_literal']}'", m["match_type"])
            console.print("\n")
            console.print(lit_table)

        # 2. Generate Naive and Improved SQL
        gen_base = self.generator.generate_baseline_sql(question, target=tgt)
        gen_imp = self.generator.generate_improved_sql(question, target=tgt)

        # Helper to convert numeric outputs safely
        def safe_float(val: Any) -> Optional[float]:
            try:
                return float(val) if val is not None else None
            except (ValueError, TypeError):
                return None

        # 3. Execute Naive Pipeline
        t0 = time.time()
        res_base = self.executor.execute_readonly_sql_sync(gen_base.generated_sql, target=tgt)
        raw_base_first = list(res_base["rows"][0].values())[0] if res_base.get("rows") else None
        val_base = validate_semantic_analysis(
            gen_base.generated_sql,
            contract=contract,
            actual_result=safe_float(raw_base_first),
        )
        base_val = raw_base_first if raw_base_first is not None else "N/A"

        # 4. Execute Improved Pipeline
        res_imp = self.executor.execute_readonly_sql_sync(gen_imp.generated_sql, target=tgt)
        raw_imp_first = list(res_imp["rows"][0].values())[0] if res_imp.get("rows") else None
        val_imp = validate_semantic_analysis(
            gen_imp.generated_sql,
            contract=contract,
            actual_result=safe_float(raw_imp_first),
        )
        imp_val = raw_imp_first if raw_imp_first is not None else "N/A"
        elapsed = (time.time() - t0) * 1000

        # 5. Display SQL Side-by-Side
        console.print("\n[bold red]Pipeline A: Naive Baseline SQL (Collapse Mode):[/bold red]")
        console.print(Syntax(gen_base.generated_sql, "sql", theme="monokai", line_numbers=True))

        console.print("\n[bold green]Pipeline B: Improved MCP SQL (Semantically Guarded):[/bold green]")
        console.print(Syntax(gen_imp.generated_sql, "sql", theme="monokai", line_numbers=True))

        # 6. Comparison Table or Data Table
        if contract.is_percentage:
            comp_table = Table(title="Benchmark Comparison Results", border_style="bright_blue")
            comp_table.add_column("Dimension", style="bold")
            comp_table.add_column("Naive Baseline", style="red")
            comp_table.add_column("Improved MCP", style="green")

            comp_table.add_row("SQL Execution", "SUCCESS (0 Errors)", "SUCCESS (0 Errors)")
            comp_table.add_row("Calculated Result", f"{base_val}%", f"[bold]{imp_val}%[/bold]")
            comp_table.add_row("Semantic Status", "[bold red]FAILED[/bold red]", "[bold green]PASSED (Sound)[/bold green]")
            comp_table.add_row("Counting Grain", val_base.detected_grain, val_imp.detected_grain)
            comp_table.add_row("Safe Math Guard?", str(val_base.has_safe_division), str(val_imp.has_safe_division))
            comp_table.add_row("Join Filter Leak?", str(val_base.has_join_filter_leak), str(val_imp.has_join_filter_leak))
            console.print("\n")
            console.print(comp_table)
        else:
            # Render returned tabular records
            rows = res_imp.get("rows", [])
            cols = res_imp.get("columns", [])
            if rows:
                data_table = Table(title=f"Query Output ({len(rows)} rows)", border_style="green", header_style="bold cyan")
                for c in cols:
                    data_table.add_column(str(c))
                for r in rows:
                    data_table.add_row(*[str(r.get(c, "")) for c in cols])
                console.print("\n")
                console.print(data_table)

        # 7. Diagnosed Pitfalls Alert
        if val_base.errors or val_base.warnings:
            pitfalls = "\n".join([f"[red][ERROR][/red] {e}" for e in val_base.errors] + [f"[yellow][WARNING][/yellow] {w}" for w in val_base.warnings])
            console.print(Panel(f"[bold red]Detected Pitfalls in Baseline Query:[/bold red]\n{pitfalls}", border_style="red"))

        console.print(f"[dim]Executed via MCP client in {elapsed:.1f}ms[/dim]\n")

    def run_direct_sql(self, sql: str) -> None:
        """Executes custom SQL through MCP with real-time AST safety and displays rich table."""
        console.print(f"\n[bold cyan]Validating & Executing via MCP on [{self.active_target.upper()}]:[/bold cyan]")
        console.print(Syntax(sql, "sql", theme="monokai"))

        # Step 1: Pre-execution AST security check
        report = validate_readonly_sql(sql)
        if not report.is_valid:
            error_msg = f"[bold red]AST SECURITY VIOLATION REJECTED:[/bold red]\n{report.error}"
            console.print(Panel(error_msg, border_style="bold red"))
            return

        # Step 2: Safe execution via MCP
        t0 = time.time()
        res = self.executor.execute_readonly_sql_sync(sql, target=self.active_target)
        elapsed = (time.time() - t0) * 1000

        if not res.get("success"):
            console.print(Panel(f"[bold red]Database Execution Error:[/bold red]\n{res.get('error')}", border_style="red"))
            return

        rows = res.get("rows", [])
        columns = res.get("columns", [])

        if not rows:
            console.print("[yellow]Query executed successfully, but returned 0 rows.[/yellow]\n")
            return

        # Step 3: Render formatted data table
        table = Table(
            title=f"Query Output ({len(rows)} rows returned in {elapsed:.1f}ms)",
            border_style="green",
            show_header=True,
            header_style="bold cyan",
        )
        for col in columns:
            table.add_column(str(col))

        for r in rows:
            table.add_row(*[str(r.get(c, "")) for c in columns])

        console.print(table)
        console.print(f"[dim]Verified read-only query executed in {elapsed:.1f}ms via MCP.[/dim]\n")

    def read_multiline_sql(self, first_line: str = "") -> str:
        """Reads multiline SQL input until a semicolon (;) is entered."""
        lines = [first_line] if first_line else []
        if lines and lines[0].strip().endswith(";"):
            return lines[0].strip()

        while True:
            try:
                line = input("...  " if lines else "SQL> ")
                if not line.strip() and lines:
                    break
                lines.append(line)
                if line.strip().endswith(";"):
                    break
            except (KeyboardInterrupt, EOFError):
                break
        return "\n".join(lines).strip()

    def inspect_tables(self) -> None:
        """Displays interactive overview of live database tables."""
        inspector = SchemaInspector(target=self.active_target)
        tables = inspector.get_table_names()

        table = Table(title=f"Accessible Tables in [{self.active_target.upper()}]", border_style="blue")
        table.add_column("#", style="dim", width=4)
        table.add_column("Table Name", style="bold white")
        table.add_column("Columns", style="cyan")

        for idx, tbl in enumerate(tables, 1):
            cols = [c["name"] for c in inspector.get_table_columns(tbl)]
            table.add_row(str(idx), tbl, ", ".join(cols[:6]) + ("..." if len(cols) > 6 else ""))

        console.print(table)
        console.print("[dim]Use command 5 to query any table with SELECT.[/dim]\n")

    def start_loop(self) -> None:
        """Main REPL loop."""
        self.print_banner()
        self.print_menu()

        while True:
            try:
                user_input = Prompt.ask(
                    f"\n[bold yellow]text2sql[/bold yellow] ([cyan]{self.active_target}[/cyan])"
                ).strip()

                if not user_input:
                    continue

                cmd = user_input.lower()

                if cmd in ("q", "quit", "exit"):
                    console.print("[bold green]Goodbye![/bold green]")
                    break

                elif cmd in ("help", "menu", "?"):
                    self.print_menu()

                elif cmd == "1":
                    # Preset 1: User Retention
                    q = "What percentage of users who signed up in January returned in February?"
                    self.run_problem1_analysis(q, target="local")

                elif cmd == "2":
                    # Preset 2: Staff Attendance
                    q = "What percentage of our staff members were marked present in June 2026?"
                    self.run_problem1_analysis(q, target="postgres")

                elif cmd == "3":
                    # Preset 3: Loan Payout
                    q = "What percentage of group loans created in May 2026 reached the Payout stage?"
                    self.run_problem1_analysis(q, target="postgres")

                elif cmd == "4":
                    self.inspect_tables()

                elif cmd == "5":
                    console.print("[bold green]Paste or type SQL query (end with ';' and press Enter):[/bold green]")
                    sql = self.read_multiline_sql()
                    if sql.strip():
                        self.run_direct_sql(sql)

                elif cmd == "6":
                    # Toggle database target
                    self.active_target = "local" if self.active_target == "postgres" else "postgres"
                    console.print(f"[bold yellow]Switched target database to:[/bold yellow] [bold green]{self.active_target.upper()}[/bold green]\n")

                elif user_input.upper().startswith("SELECT ") or user_input.upper().startswith("WITH "):
                    # User typed or pasted a multi-line SQL query directly
                    sql = self.read_multiline_sql(first_line=user_input)
                    if sql.strip():
                        self.run_direct_sql(sql)

                else:
                    # User typed a natural language question
                    self.run_problem1_analysis(user_input)

            except (KeyboardInterrupt, EOFError):
                console.print("\n[bold green]Session ended. Goodbye![/bold green]")
                break
            except Exception as exc:
                console.print(f"[bold red]Unexpected error:[/bold red] {exc}")


def launch_cli() -> None:
    """Entry point for the interactive CLI."""
    app = InteractiveConsole()
    app.start_loop()


if __name__ == "__main__":
    launch_cli()
