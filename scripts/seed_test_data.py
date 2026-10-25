"""Seed test dataset for Problem 1: Cohort Retention Benchmark.

Creates the standard benchmark schema with:
- users: 100 January sign-ups, plus boundary sign-ups in Dec/Feb.
- activity_events: 30 distinct returning users with multiple events, duplicates, and NULLs.

Known Ground Truth:
- January Sign-ups (Denominator): 100
- Distinct Returning in February (Numerator): 30
- True Retention Rate: 30.0%
"""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings


def seed_local_benchmark_db(db_path: Optional[str] = None) -> None:
    """Populates the local database with deterministic benchmark data."""
    settings = get_settings()
    target_path = Path(db_path or settings.local_db_path)
    
    print(f"Seeding local benchmark database at: {target_path}")
    
    # Remove existing local database to ensure a clean, deterministic state
    if target_path.exists():
        target_path.unlink()

    conn = sqlite3.connect(str(target_path))
    cur = conn.cursor()

    # 1. Create tables
    cur.execute("""
    CREATE TABLE users (
        user_id INTEGER PRIMARY KEY,
        username TEXT NOT NULL,
        signup_date DATE NOT NULL,
        country TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE activity_events (
        event_id INTEGER PRIMARY KEY,
        user_id INTEGER NOT NULL,
        event_type TEXT NOT NULL,
        event_date DATE NOT NULL,
        device TEXT,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)

    # 2. Insert Users:
    # - Users 1..100: Signed up in January 2024 (Cohort of interest)
    # - Users 101..110: Signed up in December 2023 (Prior boundary check)
    # - Users 111..120: Signed up in February 2024 (Post boundary check)
    users_data = []
    
    # Prior boundary (December 2023)
    for u in range(101, 111):
        users_data.append((u, f"user_{u}", "2023-12-25", "US"))

    # The 100 January 2024 sign-ups
    for u in range(1, 101):
        # Distribute signups across January (Jan 1 to Jan 31)
        day = (u % 31) + 1
        day_str = f"2024-01-{day:02d}"
        country = "US" if u % 2 == 0 else "CA"
        users_data.append((u, f"user_{u}", day_str, country))

    # Post boundary (February 2024)
    for u in range(111, 121):
        users_data.append((u, f"user_{u}", "2024-02-05", "UK"))

    cur.executemany("INSERT INTO users VALUES (?, ?, ?, ?);", users_data)

    # 3. Insert Activity Events:
    # Exactly 30 distinct January users (users 1..30) return in February 2024.
    # Users 1..30 generate multiple events each (simulating repeat sessions).
    # Users 31..100 have NO events in February (churned).
    events_data = []
    event_id = 1

    # January users returning in February
    for u in range(1, 31):
        # Vary repeat events: User 1 has 6 events, User 2 has 5, etc.
        num_events = (u % 5) + 2
        for e_idx in range(num_events):
            day = (e_idx * 4) + 1
            event_date = f"2024-02-{day:02d}"
            device = "Mobile" if e_idx % 2 == 0 else None  # Includes NULLs
            events_data.append((event_id, u, "login", event_date, device))
            event_id += 1

    # Include some events outside February for users 1..30 (e.g. in March)
    for u in range(1, 15):
        events_data.append((event_id, u, "purchase", "2024-03-02", "Desktop"))
        event_id += 1

    # Include February events for December/February signups (users 101..120)
    # These must NOT be counted in the January cohort retention!
    for u in range(101, 115):
        events_data.append((event_id, u, "login", "2024-02-14", "Mobile"))
        event_id += 1

    cur.executemany("INSERT INTO activity_events VALUES (?, ?, ?, ?, ?);", events_data)
    conn.commit()

    # 4. Verify Ground Truth with deterministic verification query
    cur.execute("""
    SELECT 
        COUNT(DISTINCT u.user_id) AS total_jan_signups,
        COUNT(DISTINCT CASE 
            WHEN e.event_date >= '2024-02-01' AND e.event_date < '2024-03-01' THEN u.user_id 
            ELSE NULL 
        END) AS returning_feb_users,
        ROUND(
            (COUNT(DISTINCT CASE 
                WHEN e.event_date >= '2024-02-01' AND e.event_date < '2024-03-01' THEN u.user_id 
                ELSE NULL 
            END) * 100.0) / COUNT(DISTINCT u.user_id), 
            2
        ) AS true_retention_rate
    FROM users u
    LEFT JOIN activity_events e ON u.user_id = e.user_id
    WHERE u.signup_date >= '2024-01-01' AND u.signup_date < '2024-02-01';
    """)
    jan_signups, returning_users, retention_rate = cur.fetchone()

    cur.execute("SELECT COUNT(*) FROM activity_events WHERE event_date >= '2024-02-01' AND event_date < '2024-03-01';")
    total_feb_events = cur.fetchone()[0]

    conn.close()

    print("[SUCCESS] Local benchmark database created successfully!")
    print(f"  - Total users: {len(users_data)}")
    print(f"  - Total activity events: {len(events_data)}")
    print(f"  - Total February events: {total_feb_events}")
    print(f"  - Ground Truth Jan Signups (Denominator): {jan_signups}")
    print(f"  - Ground Truth Returning Users in Feb (Numerator): {returning_users}")
    print(f"  - Ground Truth Retention Rate: {retention_rate}%\n")


if __name__ == "__main__":
    seed_local_benchmark_db()
