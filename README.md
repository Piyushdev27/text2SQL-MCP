# Text-to-SQL Problem 1: Erroneous Data Analysis Using MCP and PostgreSQL

An enterprise-grade Text-to-SQL system built from scratch using the **Model Context Protocol (MCP)** and **PostgreSQL**, dedicated exclusively to solving **Problem 1: Erroneous Data Analysis**.

---

## 1. Problem 1 Overview: Erroneous Data Analysis

In the Spider 2.0 benchmark, **35.5% of real-world text-to-SQL failures** stem from *Erroneous Data Analysis*.

### Core Concept: Execution Success != Semantic Correctness
A query can be syntactically valid in PostgreSQL and execute with zero errors (`HTTP 200 / Status SUCCESS`), yet produce a completely incorrect business answer due to:
1. **Grain & Counting Mismatches**: Counting row/event occurrences (`COUNT(*)`, `COUNT(event_id)`) instead of distinct entities (`COUNT(DISTINCT user_id)`).
2. **Filter Misplacement**: Placing date predicates on a `LEFT JOIN` in the `WHERE` clause, inadvertently turning it into an `INNER JOIN` and dropping the churned cohort.
3. **Denominator Distortion**: Dividing returning users by active users or activity events instead of the baseline signup cohort.
4. **Division-by-Zero Crashes**: Dividing without guarding the denominator with `NULLIF(col, 0)`.

---

## 2. Architecture

```
User Question ("What % of Jan signups returned in Feb?")
         │
         ▼
┌────────────────────────┐
│    SQL Generator       │  Decomposes question into AnalyticalContract
└────────┬───────────────┘
         │  Requests schema cards & sample rows via JSON-RPC
         ▼
┌────────────────────────┐
│   MCP Query Executor   │  Official Python MCP Client layer
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│       MCP Server       │  Exposes 5 safe tools:
│   (MCPServer v2.3)     │  1. list_tables
└────────┬───────────────┘  2. get_table_schema
         │                  3. get_sample_rows
         │                  4. validate_sql (AST inspection)
         │                  5. execute_readonly_sql (timeout + row limits)
         ▼
┌────────────────────────┐
│  PostgreSQL / Local DB │  Enforces read-only transactions & 60s statement timeout
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐
│   Semantic Validator   │  AST inspection:
│   (AST + Result check) │  - Enforces COUNT(DISTINCT entity)
└────────┬───────────────┘  - Detects LEFT JOIN filter leaks
         │                  - Verifies plausibility bounds (0% <= rate <= 100%)
         ▼
Verified Business Answer (30.0% Retention Rate)
```

---

## 3. Project Structure

```
text_to_sql/
├── src/
│   ├── __init__.py            # Package root (version 0.1.0)
│   ├── config.py              # Pydantic Settings with masked credentials and .env loading
│   ├── db.py                  # PostgreSQL connection factory with 60s timeout & read-only session
│   ├── schema_inspector.py    # Generates LLM schema cards and fetches bounded sample rows
│   ├── sql_validator.py       # SQLGlot AST validator rejecting DDL, DML, and multi-statements
│   ├── mcp_server.py          # Official Anthropic MCP Server with 5 database tools
│   ├── query_executor.py      # MCP Client layer wrapping tool discovery and invocation
│   ├── semantic_validator.py  # Diagnoses business logic pitfalls (grain, join traps, bounds)
│   ├── sql_generator.py       # Generates baseline (naive) and improved (guarded) SQL
│   └── evaluator.py           # Calculates dynamic ground truth and compares pipeline accuracy
├── scripts/
│   ├── inspect_database.py    # CLI tool to inspect tables and columns in PostgreSQL
│   ├── seed_test_data.py      # Populates the 120-user January/February retention benchmark
│   └── run_demo.py            # End-to-end demonstration comparing Baseline vs Improved
├── tests/
│   ├── __init__.py
│   ├── test_db_connection.py  # Tests live PostgreSQL connectivity and session settings
│   ├── test_mcp_tools.py      # Tests discovery and execution of all 5 MCP tools
│   ├── test_sql_validator.py  # Tests AST rejection of DROP, DELETE, UPDATE, and multi-statements
│   └── test_semantic_analysis.py # Tests retention accuracy, grain handling, and join leak detection
├── .env.example               # Sanitized template with configuration placeholders
├── .gitignore                 # Excludes .env, __pycache__, local databases, and logs
├── requirements.txt           # Minimal dependencies (mcp, psycopg2-binary, sqlglot, pydantic, pytest)
└── README.md                  # Project documentation and architectural guide
```

---

## 4. Setup and Installation

### 4.1 Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Access to PostgreSQL or local test SQLite database

### 4.2 Install Dependencies
```bash
pip install -r requirements.txt
```

### 4.3 Configure Environment Variables
Copy `.env.example` to `.env` and fill in your PostgreSQL credentials:
```bash
cp .env.example .env
```
Example `.env`:
```ini
TEXT2SQL_POSTGRES_HOST=35.200.180.162
TEXT2SQL_POSTGRES_PORT=5432
TEXT2SQL_POSTGRES_USER=t2sql_reader
TEXT2SQL_POSTGRES_PASSWORD=your_password_here
TEXT2SQL_POSTGRES_DB=lending_sample
TEXT2SQL_POSTGRES_SSLMODE=require
TEXT2SQL_POSTGRES_SCHEMA=public
```

---

## 5. Execution Commands

### 1. Inspect the Live PostgreSQL Database
```bash
python scripts/inspect_database.py
```
Outputs discovered tables (`group_loans`, `individual_loans`, `outlets`, `staff`, etc.), column types, nullability, and row counts.

### 2. Seed the Problem 1 Benchmark Test Dataset
```bash
python scripts/seed_test_data.py
```
Creates `local_problem1.db` containing:
- 100 January 2024 sign-ups (Cohort denominator).
- 30 distinct January users returning in February (Cohort numerator).
- 134 February activity events (repeat logins, testing grain mismatch).
- Known mathematical ground truth: **30.0%**.

### 3. Run the End-to-End Baseline vs. Improved Demo
```bash
python scripts/run_demo.py
```

### 4. Run the Automated Test Suite
```bash
python -m pytest tests/ -v
```
Runs 25 automated tests verifying connectivity, MCP tools, SQL AST security, and semantic analysis accuracy.

---

## 6. Demonstration Summary: Baseline vs. Improved

| Dimension | Pipeline A (Naive Baseline) | Pipeline B (Improved MCP + Semantic) |
| :--- | :--- | :--- |
| **Generated SQL** | `COUNT(e.event_id) * 100.0 / COUNT(u.user_id)` | `(COUNT(DISTINCT CASE WHEN e.event_date ... THEN u.user_id END) * 100.0) / NULLIF(COUNT(DISTINCT u.user_id), 0)` |
| **SQL Execution** | SUCCESS (0 errors) | SUCCESS (0 errors) |
| **Returned Result** | **100.0%** | **30.0%** |
| **Ground Truth** | **30.0%** | **30.0%** |
| **Absolute Error** | **70.00%** (Failure) | **0.00%** (Exact Match) |
| **Semantic Check** | FAILED (Grain Mismatch, Filter Leak) | PASSED (Semantically Sound) |
