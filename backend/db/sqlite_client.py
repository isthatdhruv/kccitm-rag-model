"""SQLite helper — async (aiosqlite) for FastAPI, sync init for CLI.

Creates ALL tables for the entire project upfront so we never need migrations.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import aiosqlite

from config import settings

# ---------------------------------------------------------------------------
# Table definitions keyed by config attribute name → list of DDL statements
# ---------------------------------------------------------------------------

_TABLES: dict[str, list[str]] = {
    "SESSION_DB": [
        """CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT CHECK(role IN ('user','assistant','system')),
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'faculty' CHECK(role IN ('admin','faculty')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
    ],
    "CACHE_DB": [
        """CREATE TABLE IF NOT EXISTS query_cache (
            id TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            query_hash TEXT NOT NULL,
            query_embedding BLOB,
            response TEXT NOT NULL,
            route_used TEXT,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hit_count INTEGER DEFAULT 0,
            last_hit_at TIMESTAMP
        )""",
        "CREATE INDEX IF NOT EXISTS idx_cache_hash ON query_cache(query_hash)",
    ],
    "FEEDBACK_DB": [
        """CREATE TABLE IF NOT EXISTS feedback (
            id TEXT PRIMARY KEY,
            message_id TEXT,
            session_id TEXT,
            query_text TEXT NOT NULL,
            response_text TEXT NOT NULL,
            rating INTEGER,
            feedback_text TEXT,
            implicit_signals TEXT,
            route_used TEXT,
            sql_generated TEXT,
            chunks_used TEXT,
            reranker_scores TEXT,
            confidence_score REAL,
            quality_score REAL,
            healed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS implicit_signals (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            signal_type TEXT,
            original_query TEXT,
            follow_up_query TEXT,
            time_gap_seconds INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS chunk_analytics (
            chunk_id TEXT PRIMARY KEY,
            times_retrieved INTEGER DEFAULT 0,
            times_reranked_top5 INTEGER DEFAULT 0,
            times_in_final_context INTEGER DEFAULT 0,
            avg_reranker_score REAL DEFAULT 0.0,
            avg_query_quality_score REAL DEFAULT 0.0,
            last_retrieved_at TIMESTAMP,
            never_retrieved INTEGER DEFAULT 1
        )""",
        """CREATE TABLE IF NOT EXISTS training_candidates (
            id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            quality_score REAL,
            category TEXT,
            source TEXT,
            included_in_training INTEGER DEFAULT 0,
            training_run_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
    ],
    "PROMPTS_DB": [
        """CREATE TABLE IF NOT EXISTS prompt_templates (
            id TEXT PRIMARY KEY,
            prompt_name TEXT NOT NULL,
            section_name TEXT NOT NULL,
            content TEXT NOT NULL,
            version INTEGER DEFAULT 1,
            is_active INTEGER DEFAULT 1,
            performance_score REAL,
            query_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS prompt_evolution_log (
            id TEXT PRIMARY KEY,
            prompt_name TEXT,
            section_name TEXT,
            old_version INTEGER,
            new_version INTEGER,
            change_reason TEXT,
            change_diff TEXT,
            approved_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
        """CREATE TABLE IF NOT EXISTS faq_entries (
            id TEXT PRIMARY KEY,
            canonical_question TEXT,
            answer TEXT NOT NULL,
            source_queries TEXT,
            avg_quality_score REAL,
            hit_count INTEGER DEFAULT 0,
            last_hit_at TIMESTAMP,
            status TEXT DEFAULT 'active',
            admin_verified INTEGER DEFAULT 0,
            data_version TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""",
    ],
}


# ---------------------------------------------------------------------------
# Sync initialiser (for CLI)
# ---------------------------------------------------------------------------


def init_all_dbs(cfg=None) -> None:
    """Create all .db files and all tables. Safe to call repeatedly."""
    if cfg is None:
        cfg = settings

    GREEN = "\033[92m"
    RESET = "\033[0m"

    for attr, ddl_list in _TABLES.items():
        db_rel = getattr(cfg, attr)
        db_path = cfg.db_path(db_rel)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        for ddl in ddl_list:
            conn.execute(ddl)
        conn.commit()
        conn.close()
        print(f"{GREEN}✓ Initialised {db_path.name}{RESET}")


# ---------------------------------------------------------------------------
# Async helpers (for FastAPI)
# ---------------------------------------------------------------------------


async def execute(db_path: str | Path, sql: str, params: tuple | None = None) -> None:
    """Execute a write statement (INSERT/UPDATE/DELETE)."""
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute(sql, params or ())
        await db.commit()


async def fetch_one(db_path: str | Path, sql: str, params: tuple | None = None) -> dict | None:
    """Return a single row as a dict, or None."""
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params or ()) as cur:
            row = await cur.fetchone()
            return dict(row) if row else None


async def fetch_all(db_path: str | Path, sql: str, params: tuple | None = None) -> list[dict]:
    """Return all rows as a list of dicts."""
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(sql, params or ()) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]


async def insert(db_path: str | Path, sql: str, params: tuple | None = None) -> int:
    """Insert a row and return lastrowid."""
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        cur = await db.execute(sql, params or ())
        await db.commit()
        return cur.lastrowid
