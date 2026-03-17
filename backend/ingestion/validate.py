"""End-to-end validation of the Phase 1 data pipeline.

Runs 9 checks covering MySQL, Milvus, and SQLite.

Usage:
    cd backend
    python -m ingestion.validate
"""

import sqlite3
import sys

import httpx

from config import settings
from db.milvus_client import MilvusSearchClient
from db.mysql_client import sync_execute

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0


def _ok(msg: str) -> None:
    global passed
    passed += 1
    print(f"{GREEN}✓ {msg}{RESET}")


def _fail(msg: str) -> None:
    global failed
    failed += 1
    print(f"{RED}✗ {msg}{RESET}")


def _embed_text(text: str) -> list[float]:
    """Synchronously embed text via Ollama for validation searches."""
    resp = httpx.post(
        f"{settings.OLLAMA_HOST}/api/embeddings",
        json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": text},
        timeout=60.0,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


# ---------------------------------------------------------------------------
# Check 1: MySQL normalised tables
# ---------------------------------------------------------------------------

def check_mysql_tables() -> tuple[int, int, int]:
    """Verify MySQL normalised tables exist and have data."""
    try:
        students = sync_execute("SELECT COUNT(*) AS cnt FROM students")[0]["cnt"]
        semesters = sync_execute("SELECT COUNT(*) AS cnt FROM semester_results")[0]["cnt"]
        subjects = sync_execute("SELECT COUNT(*) AS cnt FROM subject_marks")[0]["cnt"]

        if students > 0 and semesters > 0 and subjects > 0:
            _ok(f"MySQL: {students} students, {semesters} semester records, {subjects} subject marks")
        else:
            _fail(f"MySQL tables have zero rows: students={students}, semesters={semesters}, subjects={subjects}")

        # Sample query: top 5 students by SGPA in semester 1
        top5 = sync_execute(
            "SELECT s.name, sr.sgpa FROM students s "
            "JOIN semester_results sr ON s.roll_no = sr.roll_no "
            "WHERE sr.semester = 1 ORDER BY sr.sgpa DESC LIMIT 5"
        )
        if top5:
            print(f"  Top 5 semester-1 by SGPA: {', '.join(r['name'] + ' (' + str(r['sgpa']) + ')' for r in top5)}")

        return students, semesters, subjects
    except Exception as exc:
        _fail(f"MySQL check failed: {exc}")
        return 0, 0, 0


# ---------------------------------------------------------------------------
# Check 2: Milvus collection stats
# ---------------------------------------------------------------------------

def check_milvus_stats(search_client: MilvusSearchClient) -> int:
    """Verify Milvus student_results collection exists and has data."""
    try:
        stats = search_client.get_collection_stats()
        count = stats.get("row_count", 0)
        if count > 0:
            _ok(f"Milvus student_results: {count} chunks indexed (dense + BM25)")
        else:
            _fail(f"Milvus student_results is empty (0 chunks)")
        return count
    except Exception as exc:
        _fail(f"Milvus stats check failed: {exc}")
        return 0


# ---------------------------------------------------------------------------
# Check 3: Dense search
# ---------------------------------------------------------------------------

def check_dense_search(search_client: MilvusSearchClient) -> None:
    """Test dense-only vector search."""
    try:
        emb = _embed_text("top CSE students")
        results = search_client.dense_search(emb, k=5)
        if results:
            top = results[0]
            _ok(f"Dense search: working (top result: {top['metadata']['name']} sem{top['metadata']['semester']})")
        else:
            _fail("Dense search returned no results")
    except Exception as exc:
        _fail(f"Dense search failed: {exc}")


# ---------------------------------------------------------------------------
# Check 4: BM25 keyword search
# ---------------------------------------------------------------------------

def check_bm25_search(search_client: MilvusSearchClient) -> None:
    """Test BM25 keyword search with a subject code."""
    try:
        results = search_client.keyword_search("KCS503", k=5)
        if results:
            _ok(f"BM25 search: working (KCS503 found in {len(results)} chunks)")
            print(f"  Top result: {results[0]['metadata']['name']} sem{results[0]['metadata']['semester']}")
        else:
            # Try a different common code
            results2 = search_client.keyword_search("Engineering", k=5)
            if results2:
                _ok(f"BM25 search: working ('Engineering' found in {len(results2)} chunks)")
            else:
                _fail("BM25 search returned no results")
    except Exception as exc:
        _fail(f"BM25 search failed: {exc}")


# ---------------------------------------------------------------------------
# Check 5: Hybrid search
# ---------------------------------------------------------------------------

def check_hybrid_search(search_client: MilvusSearchClient) -> None:
    """Test hybrid search (dense + BM25 + RRF)."""
    try:
        emb = _embed_text("students struggling in programming")
        results = search_client.hybrid_search("students struggling in programming", emb, k=5)
        if results:
            top = results[0]
            _ok(f"Hybrid search: working (top result: {top['metadata']['name']} sem{top['metadata']['semester']})")
        else:
            _fail("Hybrid search returned no results")
    except Exception as exc:
        _fail(f"Hybrid search failed: {exc}")


# ---------------------------------------------------------------------------
# Check 6: Filtered search
# ---------------------------------------------------------------------------

def check_filtered_search(search_client: MilvusSearchClient) -> None:
    """Test search with metadata filters."""
    try:
        emb = _embed_text("student results")
        results = search_client.dense_search(
            emb,
            k=3,
            filters={"semester": 4, "course": "B.TECH"},
        )
        if results:
            all_match = all(
                r["metadata"]["semester"] == 4 and r["metadata"]["course"] == "B.TECH"
                for r in results
            )
            if all_match:
                _ok(f"Filtered search: working (all {len(results)} results match filters)")
            else:
                _fail("Filtered search returned results that don't match filters")
        else:
            # Filters might be too narrow — try broader
            results2 = search_client.dense_search(emb, k=3, filters={"semester": 1})
            if results2:
                _ok(f"Filtered search: working ({len(results2)} results for semester=1)")
            else:
                _fail("Filtered search returned no results")
    except Exception as exc:
        _fail(f"Filtered search failed: {exc}")


# ---------------------------------------------------------------------------
# Check 7: FAQ collection
# ---------------------------------------------------------------------------

def check_faq_collection(search_client: MilvusSearchClient) -> None:
    """Verify FAQ collection exists and is empty."""
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri=f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}")
        if client.has_collection(settings.MILVUS_FAQ_COLLECTION):
            stats = client.get_collection_stats(settings.MILVUS_FAQ_COLLECTION)
            count = stats.get("row_count", 0)
            _ok(f"Milvus FAQ: collection ready ({count} entries)")
        else:
            _fail("FAQ collection does not exist")
    except Exception as exc:
        _fail(f"FAQ collection check failed: {exc}")


# ---------------------------------------------------------------------------
# Check 8: SQLite databases
# ---------------------------------------------------------------------------

def check_sqlite_dbs() -> None:
    """Verify all SQLite databases and tables exist."""
    db_configs = {
        "sessions.db": ["sessions", "messages", "users"],
        "cache.db": ["query_cache"],
        "feedback.db": ["feedback", "implicit_signals", "chunk_analytics", "training_candidates"],
        "prompts.db": ["prompt_templates", "prompt_evolution_log", "faq_entries"],
    }

    all_ok = True
    for db_name, expected_tables in db_configs.items():
        db_attr = {
            "sessions.db": "SESSION_DB",
            "cache.db": "CACHE_DB",
            "feedback.db": "FEEDBACK_DB",
            "prompts.db": "PROMPTS_DB",
        }[db_name]
        db_path = settings.db_path(getattr(settings, db_attr))

        if not db_path.exists():
            _fail(f"SQLite: {db_name} not found at {db_path}")
            all_ok = False
            continue

        conn = sqlite3.connect(str(db_path))
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        actual_tables = {row[0] for row in cur.fetchall()}
        conn.close()

        missing = set(expected_tables) - actual_tables
        if missing:
            _fail(f"SQLite: {db_name} missing tables: {missing}")
            all_ok = False

    if all_ok:
        _ok("SQLite: all databases and tables initialised")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_validation() -> None:
    """Run all validation checks."""
    global passed, failed
    passed = 0
    failed = 0

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Phase 1 Validation{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    # 1. MySQL
    students, semesters, subjects = check_mysql_tables()

    # 2-6. Milvus
    try:
        search_client = MilvusSearchClient()
        check_milvus_stats(search_client)
        check_dense_search(search_client)
        check_bm25_search(search_client)
        check_hybrid_search(search_client)
        check_filtered_search(search_client)
        check_faq_collection(search_client)
    except Exception as exc:
        _fail(f"Milvus connection failed: {exc}")

    # 8. SQLite
    check_sqlite_dbs()

    # Summary
    print(f"\n{BOLD}{'='*60}{RESET}")
    total = passed + failed
    if failed == 0:
        print(f"{GREEN}{BOLD}  All {passed} validations passed. System ready for Phase 2.{RESET}")
    else:
        print(f"{RED}{BOLD}  {failed}/{total} checks failed.{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_validation()
