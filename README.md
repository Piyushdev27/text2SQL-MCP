# Enterprise Text-to-SQL: Erroneous Analysis & Schema Linking using MCP and PostgreSQL

An enterprise-grade Text-to-SQL system built from scratch using the **Model Context Protocol (MCP)**, **PostgreSQL**, and **AST-based Semantic Verification**, solving:
1. **Problem 1: Erroneous Data Analysis** (Grain mismatch, filter leaks, denominator distortion, Cartesian explosion).
2. **Problem 2: Schema Linking & Large-Scale Schema Pruning** (Coarse table pruning, fine column pruning, relational graph BFS join discovery, token reduction).

---

## 1. Core Problems Solved

### Problem 1: Erroneous Data Analysis (35.5% of Spider 2.0 Failures)
A query can execute with zero SQL errors (`HTTP 200 / Status SUCCESS`), yet produce completely wrong business metrics:
- **Grain & Counting Mismatches**: Counting row occurrences (`COUNT(*)`, `COUNT(event_id)`) instead of distinct entities (`COUNT(DISTINCT user_id)`).
- **Filter Misplacement**: Placing date predicates on a `LEFT JOIN` in the `WHERE` clause, inadvertently turning it into an `INNER JOIN` and dropping the churned cohort.
- **Denominator Distortion**: Dividing returning users by active users or activity events instead of the baseline signup cohort.
- **Cartesian Fan-Out Explosion**: Joining multi-table aggregates without CTE pre-aggregation, blowing up monthly financial totals.
- **Division-by-Zero Crashes**: Dividing without guarding denominators using `NULLIF(col, 0)`.

### Problem 2: Schema Linking & Large-Scale Schema Pruning
In enterprise data warehouses, massive table counts and hundreds of audit columns cause context bloat, column hallucinations, and invalid joins:
- **Coarse Table Pruning**: Dynamic relevance scoring prioritizing grounded literals, domain tags, query stems, and bridging tables.
- **Fine Column Pruning**: Stripping irrelevant audit/timestamp columns while preserving Primary Keys, Foreign Keys, query metrics, and literal targets (e.g. **88% column reduction** on loan tables).
- **Relational Graph BFS Join Discovery**: Traverses logical knowledge graphs to discover multi-hop join paths (e.g. `staff_attendance` $\to$ `staff` $\to$ `outlets`) even when CDC mirrors omit physical FK constraints.
- **Fuzzy Literal Grounding**: Matches user typos ("Maharahtra" $\to$ "Maharashtra", "Tier 1" $\to$ "T1") against live column values.

---

## 2. Architecture & Official MCP Tools

```
User Question ("What % of Jan signups returned in Feb?")
         │
         ▼
┌────────────────────────┐
│    SQL Generator       │  Decomposes question into AnalyticalContract
└────────┬───────────────┘
         │  Requests schema discovery & literal grounding via JSON-RPC
         ▼
┌────────────────────────┐
│   MCP Query Executor   │  Official Python MCP Client layer
└────────┬───────────────┘
         │
         ▼
┌────────────────────────┐  Exposes 9 official MCP tools:
│       MCP Server       │  1. list_tables
│   (MCPServer v2.3)     │  2. get_table_schema
└────────┬───────────────┘  3. get_sample_rows
         │                  4. validate_sql (AST inspection)
         │                  5. execute_readonly_sql (timeout + row limits)
         │                  6. resolve_categorical_literals (fuzzy grounding)
         │                  7. prune_schema (coarse + fine context reduction)
         │                  8. find_join_path (graph BFS shortest path)
         │                  9. get_schema_graph (relational graph export)
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
│   ├── __init__.py            # Package root exposing LangChain tools and prompts (v0.2.0)
│   ├── config.py              # Pydantic Settings with masked credentials and .env loading
│   ├── db.py                  # PostgreSQL connection factory with 60s timeout & read-only session
│   ├── schema_inspector.py    # Generates LLM schema cards and fetches bounded sample rows
│   ├── schema_graph.py        # Problem 2 Relational Knowledge Graph & BFS shortest join paths
│   ├── schema_pruner.py       # Problem 2 Coarse Table & Fine Column Pruner (Token Reduction)
│   ├── literal_resolver.py    # Fuzzy literal grounding and database typo matching
│   ├── sql_validator.py       # SQLGlot AST validator rejecting DDL, DML, and multi-statements
│   ├── mcp_server.py          # Official Anthropic MCP Server with 9 database tools
│   ├── query_executor.py      # MCP Client layer wrapping tool discovery and invocation
│   ├── langchain_tools.py     # Native LangChain BaseTool adapters with sync/async support
│   ├── langchain_prompts.py   # ChatPromptTemplate and SQLPlan Pydantic model for structured output
│   ├── semantic_validator.py  # Diagnoses business logic pitfalls (grain, join traps, bounds)
│   ├── sql_generator.py       # Generates baseline (naive) and improved (guarded) SQL
│   ├── evaluator.py           # Calculates dynamic ground truth and compares pipeline accuracy
│   └── cli.py                 # Interactive Rich terminal REPL with Problem 1 & 2 sandboxes
├── scripts/
│   ├── inspect_database.py    # CLI tool to inspect tables and columns in PostgreSQL
│   ├── seed_test_data.py      # Populates the 120-user January/February retention benchmark
│   └── run_demo.py            # End-to-end demonstration comparing Baseline vs Improved
├── tests/
│   ├── __init__.py
│   ├── test_db_connection.py  # Tests live PostgreSQL connectivity and session settings
│   ├── test_literal_resolver.py # Tests fuzzy matching and typo resolution
│   ├── test_mcp_tools.py      # Tests discovery and execution of all 9 MCP tools
│   ├── test_schema_pruner.py  # Tests BFS join paths, table scoring, and column pruning
│   ├── test_langchain_tools.py # Tests LangChain BaseTool wrappers, LCEL prompts, and SQLPlan
│   ├── test_sql_validator.py  # Tests AST rejection of DROP, DELETE, UPDATE, and multi-statements
│   └── test_semantic_analysis.py # Tests retention accuracy, grain handling, and join leak detection
├── .env.example               # Sanitized template with configuration placeholders
├── .gitignore                 # Excludes .env, __pycache__, local databases, and logs
├── requirements.txt           # Core dependencies (mcp, psycopg2-binary, sqlglot, pydantic, langchain, pytest)
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

### 1. Launch the Interactive Terminal Console (CLI Mode)
```bash
python main.py
```
*(or `python main.py -i`)*
Launches an interactive developer REPL with:
- Natural Language Question evaluation.
- Direct SQL execution sandbox with real-time AST safety validation.
- Live PostgreSQL table exploration.
- Syntax-highlighted SQL diffs and colorized comparison tables.

### 2. Inspect the Live PostgreSQL Database
```bash
python main.py --inspect-pg
```
Outputs discovered tables (`group_loans`, `individual_loans`, `outlets`, `staff`, etc.), column types, nullability, and row counts.

### 3. Seed the Problem 1 Benchmark Test Dataset
```bash
python scripts/seed_test_data.py
```

### 4. Run the Automated One-Shot Demonstration
```bash
python main.py --demo
```

### 5. Run the Automated Test Suite
```bash
python -m pytest tests/ -v
```
Runs 42 automated tests verifying live PostgreSQL connectivity, all 9 MCP tools, LangChain BaseTool adapters, AST security, semantic analysis accuracy, and Problem 2 Schema Linking & Pruning.

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
