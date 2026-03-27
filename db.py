"""SQLite database schema and connection management for P4P."""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "p4p.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id   TEXT PRIMARY KEY,
            name       TEXT NOT NULL,
            date       TEXT,
            location   TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fighters (
            fighter_id  TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            nickname    TEXT,
            weight_class TEXT,
            height      TEXT,
            reach       TEXT,
            stance      TEXT,
            dob         TEXT,
            record      TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fights (
            fight_id      TEXT PRIMARY KEY,
            event_id      TEXT NOT NULL,
            date          TEXT,
            fighter_a_id  TEXT NOT NULL,
            fighter_b_id  TEXT NOT NULL,
            winner_id     TEXT,
            method        TEXT,
            round         INTEGER,
            time          TEXT,
            title_fight   INTEGER DEFAULT 0,
            weight_class  TEXT,
            FOREIGN KEY (event_id) REFERENCES events(event_id),
            FOREIGN KEY (fighter_a_id) REFERENCES fighters(fighter_id),
            FOREIGN KEY (fighter_b_id) REFERENCES fighters(fighter_id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS fight_stats (
            fight_id                      TEXT NOT NULL,
            fighter_id                    TEXT NOT NULL,
            knockdowns                    INTEGER,
            significant_strikes_landed    INTEGER,
            significant_strikes_attempted INTEGER,
            significant_strike_pct        REAL,
            total_strikes_landed          INTEGER,
            total_strikes_attempted       INTEGER,
            takedowns_landed              INTEGER,
            takedowns_attempted           INTEGER,
            takedown_pct                  REAL,
            submissions_attempted         INTEGER,
            reversals                     INTEGER,
            control_time_seconds          INTEGER,
            PRIMARY KEY (fight_id, fighter_id),
            FOREIGN KEY (fight_id) REFERENCES fights(fight_id),
            FOREIGN KEY (fighter_id) REFERENCES fighters(fighter_id)
        )
    """)

    # Track which events have been fully scraped for incremental updates
    cur.execute("""
        CREATE TABLE IF NOT EXISTS scrape_log (
            event_id   TEXT PRIMARY KEY,
            scraped_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
