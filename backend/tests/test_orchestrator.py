"""Orchestrator end-to-end test.

Requires: MySQL + Milvus + Ollama all running with data from Phase 1.

Usage:
  python -m tests.test_orchestrator              # Run automated tests
  python -m tests.test_orchestrator --interactive # Interactive chat mode
"""

import asyncio
import sys
import time

from config import settings
from core.llm_client import OllamaClient
from core.orchestrator import Orchestrator
from core.rag_pipeline import RAGPipeline
from core.router import QueryRouter
from core.sql_pipeline import SQLPipeline
from db.milvus_client import MilvusSearchClient

# ---------------------------------------------------------------------------
# Automated test cases: (query, expected_route, validation_fn, description)
# ---------------------------------------------------------------------------

AUTOMATED_TESTS = [
    # SQL queries
    (
        "top 3 students by SGPA in semester 1",
        "SQL",
        lambda r: r.success and r.sql_result and r.sql_result.row_count > 0,
        "SQL ranking",
    ),
    (
        "how many CSE students are there",
        "SQL",
        lambda r: r.success and len(r.response) > 10,
        "SQL count",
    ),
    # RAG queries
    (
        "tell me about roll number 2104920100002",
        "RAG",
        lambda r: r.success and len(r.response) > 50,
        "RAG student lookup",
    ),
    (
        "which students are struggling in programming subjects",
        "RAG",
        lambda r: r.success and len(r.response) > 50,
        "RAG semantic query",
    ),
    # Keyword query (BM25 should shine here)
    (
        "KCS503 results",
        "RAG",
        lambda r: r.success and len(r.response) > 20,
        "BM25 keyword query",
    ),
    # HYBRID
    (
        "analyze the CSE batch performance trend across semesters",
        "HYBRID",
        lambda r: r.success and len(r.response) > 100,
        "HYBRID analysis",
    ),
]


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def _build_orchestrator() -> Orchestrator:
    llm = OllamaClient()
    router = QueryRouter(llm)
    sql_pipeline = SQLPipeline(llm)
    milvus = MilvusSearchClient(settings.MILVUS_HOST, settings.MILVUS_PORT)
    rag_pipeline = RAGPipeline(llm, milvus)
    return Orchestrator(llm, router, sql_pipeline, rag_pipeline, milvus)


# ---------------------------------------------------------------------------
# Automated tests
# ---------------------------------------------------------------------------

async def run_automated():
    orchestrator = _build_orchestrator()

    passed = 0
    failed = 0

    for query, expected_route, validate_fn, description in AUTOMATED_TESTS:
        print(f"\n{'=' * 60}")
        print(f"Test: {description}")
        print(f"Query: \"{query}\"")

        start = time.time()
        result = await orchestrator.process_query(query)
        elapsed = (time.time() - start) * 1000

        print(f"Route: {result.route_used} (expected: {expected_route}) | Time: {elapsed:.0f}ms")

        if result.success:
            preview = result.response[:200].replace("\n", " ")
            print(f"Response: {preview}...")
        else:
            print(f"Error: {result.error}")

        ok = validate_fn(result)
        if ok:
            passed += 1
            print("\033[92m✓ PASSED\033[0m")
        else:
            failed += 1
            print("\033[91m✗ FAILED\033[0m")

    total = passed + failed
    pct = (passed / total * 100) if total > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Orchestrator: {passed}/{total} passed ({pct:.0f}%)")
    if pct >= 70:
        print("\033[92m✓ System is working end-to-end!\033[0m")
    else:
        print("\033[91m✗ Below 70% threshold\033[0m")


# ---------------------------------------------------------------------------
# Interactive chat mode
# ---------------------------------------------------------------------------

async def run_interactive():
    """Interactive chat mode — type questions and see live answers."""
    orchestrator = _build_orchestrator()
    chat_history: list[dict] = []

    print("\n\033[94m" + "=" * 60)
    print("KCCITM AI Assistant — Interactive Test Mode")
    print("Type your questions. Type 'quit' to exit.")
    print("=" * 60 + "\033[0m\n")

    while True:
        try:
            query = input("\033[96mYou: \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        start = time.time()
        result = await orchestrator.process_query(query, chat_history)
        elapsed = (time.time() - start) * 1000

        if result.success:
            print(f"\n\033[93m[{result.route_used} | {elapsed:.0f}ms]\033[0m")
            print(f"\033[97m{result.response}\033[0m\n")

            # Update chat history
            chat_history.append({"role": "user", "content": query})
            chat_history.append({"role": "assistant", "content": result.response})
        else:
            print(f"\n\033[91mError: {result.error}\033[0m\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--interactive" in sys.argv:
        asyncio.run(run_interactive())
    else:
        asyncio.run(run_automated())
