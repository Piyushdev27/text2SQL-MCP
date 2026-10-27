"""Categorical Literal Grounding module (Problem 2 Slice).

Solves the Entity Literal Mismatch problem where natural language mentions
(e.g., "casual leave", "Tier 1", "present") do not match raw database cell values
(e.g., 'CL', 'T1', 'PR'), causing valid SQL queries to silently return 0 rows.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field

from src.db import execute_readonly_query, get_local_db_connection

logger = logging.getLogger(__name__)

# Known domain abbreviation mappings for lending_sample dataset
COMMON_DOMAIN_SYNONYMS: Dict[str, Dict[str, str]] = {
    "staff_attendance": {
        "present": "PR",
        "pres": "PR",
        "absent": "AB",
        "casual leave": "CL",
        "sick leave": "SL",
        "week off": "WO",
        "weekly off": "WO",
        "half day": "HD",
        "public holiday": "PH",
        "loss of pay": "LWP",
    },
    "outlets": {
        "tier 1": "T1",
        "tier 2": "T2",
        "tier 3": "T3",
        "tier-1": "T1",
        "tier-2": "T2",
        "tier-3": "T3",
    },
    "group_loans": {
        "training": "Group Training",
        "group training": "Group Training",
        "payout": "Payout",
        "paid": "Payout",
        "disbursed": "Payout",
        "ready": "Ready",
        "enrolment": "Enrolment",
        "enrollment": "Enrolment",
    },
}

# Standard geographical states for typo tolerance and out-of-domain entity recognition
KNOWN_INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi"
]

STOP_WORDS = {"in", "the", "located", "at", "for", "all", "show", "of", "and", "tier", "outlets", "outlet", "by"}

# Targeted categorical columns for live inspection
INDEXED_CATEGORICAL_COLUMNS = [
    ("outlets", "state_name"),
    ("outlets", "tier"),
    ("staff_attendance", "attendance_code"),
    ("group_loans", "stage"),
    ("group_loans", "origin_type"),
    ("staff", "position"),
]


class LiteralMatch(BaseModel):
    """Represents a grounded mapping from a natural language term to an exact database literal."""

    term: str = Field(description="The user's natural language word or phrase")
    table_name: str = Field(description="Target table where the literal belongs")
    column_name: str = Field(description="Target column in the table")
    database_literal: str = Field(description="Exact case-sensitive database cell literal")
    match_type: str = Field(description="'exact', 'case_insensitive', 'synonym', or 'substring'")
    sql_predicate: str = Field(description="Ready-to-use SQL predicate, e.g. attendance_code = 'CL'")


class LiteralResolver:
    """Discovers and resolves categorical literals to prevent zero-row query collapse."""

    def __init__(self, target: str = "postgres"):
        self.target = target.lower()
        self._cache: Dict[Tuple[str, str], List[str]] = {}

    def get_distinct_column_literals(self, table_name: str, column_name: str, limit: int = 50) -> List[str]:
        """Fetches distinct cell values for a column from the active database."""
        cache_key = (table_name, column_name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self.target == "local":
            conn = get_local_db_connection()
            cur = conn.cursor()
            cur.execute(
                f'SELECT DISTINCT "{column_name}" FROM "{table_name}" WHERE "{column_name}" IS NOT NULL LIMIT {limit};'
            )
            values = [str(r[0]) for r in cur.fetchall() if r[0] is not None]
            conn.close()
            self._cache[cache_key] = values
            return values

        # Live PostgreSQL
        sql = (
            f'SELECT DISTINCT "{column_name}" FROM "{table_name}" '
            f'WHERE "{column_name}" IS NOT NULL ORDER BY "{column_name}" LIMIT {limit};'
        )
        res = execute_readonly_query(sql)
        values = [str(r[column_name]) for r in res.get("rows", []) if r.get(column_name) is not None]
        self._cache[cache_key] = values
        return values

    def resolve_literals(self, user_text: str) -> List[LiteralMatch]:
        """Scans user text and returns grounded database literals."""
        q_lower = user_text.lower()
        matches: List[LiteralMatch] = []
        matched_keys: Set[Tuple[str, str, str]] = set()

        # Step 1: Check Domain Synonyms / Abbreviations
        for table, synonym_dict in COMMON_DOMAIN_SYNONYMS.items():
            for phrase, literal in synonym_dict.items():
                # Word boundary search for phrase in query
                if re.search(r"\b" + re.escape(phrase) + r"\b", q_lower):
                    col = "attendance_code" if table == "staff_attendance" else ("tier" if table == "outlets" else "stage")
                    key = (table, col, literal)
                    if key not in matched_keys:
                        matched_keys.add(key)
                        matches.append(
                            LiteralMatch(
                                term=phrase,
                                table_name=table,
                                column_name=col,
                                database_literal=literal,
                                match_type="synonym",
                                sql_predicate=f"{col} = '{literal}'",
                            )
                        )

        # Step 2: Check Live Database Distinct Cell Values
        for table, col in INDEXED_CATEGORICAL_COLUMNS:
            try:
                db_values = self.get_distinct_column_literals(table, col)
            except Exception:
                continue

            for val in db_values:
                val_lower = val.lower()
                # Check if database value appears directly in user text
                if re.search(r"\b" + re.escape(val_lower) + r"\b", q_lower):
                    key = (table, col, val)
                    if key not in matched_keys:
                        matched_keys.add(key)
                        matches.append(
                            LiteralMatch(
                                term=val,
                                table_name=table,
                                column_name=col,
                                database_literal=val,
                                match_type="exact" if val in user_text else "case_insensitive",
                                sql_predicate=f"{col} = '{val}'",
                            )
                        )

        # Step 3: Geographical & Categorical Typo / Fuzzy Matching
        matched_state_tuple = self._resolve_state_fuzzy(user_text)
        if matched_state_tuple:
            raw_term, matched_state, match_type = matched_state_tuple
            key = ("outlets", "state_name", matched_state)
            if key not in matched_keys:
                matched_keys.add(key)
                try:
                    db_states = self.get_distinct_column_literals("outlets", "state_name")
                    if matched_state not in db_states:
                        match_type = f"{match_type} (not in DB)"
                except Exception:
                    pass

                matches.append(
                    LiteralMatch(
                        term=raw_term,
                        table_name="outlets",
                        column_name="state_name",
                        database_literal=matched_state,
                        match_type=match_type,
                        sql_predicate=f"state_name = '{matched_state}'",
                    )
                )

        return matches

    def _resolve_state_fuzzy(self, user_text: str) -> Optional[Tuple[str, str, str]]:
        """Resolves natural language states with typo-tolerance and geographical knowledge."""
        text_lower = user_text.lower()

        # 1. Check exact/case-insensitive match against known states
        for s in KNOWN_INDIAN_STATES:
            if re.search(r"\b" + re.escape(s.lower()) + r"\b", text_lower):
                return s, s, "exact"

        words = [w for w in re.findall(r"[a-zA-Z]+", user_text)]
        multi_word_states = [s for s in KNOWN_INDIAN_STATES if " " in s]
        single_word_states = [s for s in KNOWN_INDIAN_STATES if " " not in s]

        # 2. Check single words against single-word states (e.g. Maharahtra -> Maharashtra)
        for w in words:
            if w.lower() in STOP_WORDS or len(w) < 4:
                continue
            close = difflib.get_close_matches(w, single_word_states, n=1, cutoff=0.74)
            if close:
                return w, close[0], "fuzzy"

        # 3. Check bigrams against multi-word states (e.g. Utter Pradesh -> Uttar Pradesh)
        bigrams = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
        for bg in bigrams:
            close = difflib.get_close_matches(bg, multi_word_states, n=1, cutoff=0.80)
            if close:
                return bg, close[0], "fuzzy"

        return None

    def generate_grounding_prompt_hint(self, user_text: str) -> str:
        """Produces a formatted prompt hint to guide LLMs with exact database literals."""
        matches = self.resolve_literals(user_text)
        if not matches:
            return ""

        lines = ["[Discovered Database Literals (Schema Grounding)]:", "The user's query references domain terms that map to specific database cell literals:"]
        for m in matches:
            lines.append(f"  - '{m.term}' -> In table `{m.table_name}`, column `{m.column_name}`, filter with: `{m.sql_predicate}`")
        lines.append("Use these exact literals in SQL WHERE clauses instead of inventing literal values.")
        return "\n".join(lines)
