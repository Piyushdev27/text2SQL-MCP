"""Main entrypoint for Text-to-SQL Problem 1: Erroneous Data Analysis.

Demonstrates how SQL queries can execute with zero errors yet produce logically
erroneous business metrics, and shows how Model Context Protocol (MCP) and Semantic
AST Validation detect and fix these failure modes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_demo import run_demonstration
from src.cli import launch_cli


def main():
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Problem 1: Erroneous Data Analysis (MCP & PostgreSQL)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the automated one-shot baseline vs improved demo",
    )
    parser.add_argument(
        "--inspect-pg",
        action="store_true",
        help="Run full live PostgreSQL database schema inspection",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Launch the interactive terminal console (default)",
    )
    args = parser.parse_args()

    if args.inspect_pg:
        from scripts.inspect_database import inspect_postgres
        inspect_postgres()
    elif args.demo:
        run_demonstration()
    else:
        # Default behavior: Launch interactive terminal console
        launch_cli()


if __name__ == "__main__":
    main()
