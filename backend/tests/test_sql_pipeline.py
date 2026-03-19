"""SQL Pipeline end-to-end test.

Requires: MySQL running with ETL-populated tables, Ollama with qwen3:8b.
Run: python -m tests.test_sql_pipeline
"""

import asyncio

from core.llm_client import OllamaClient
from core.router import QueryRouter, RouteResult
from core.sql_pipeline import SQLPipeline

# Test cases: (query, validation_function, description)
# validation_function takes SQLResult and returns (bool, str) — (passed, reason)
TEST_CASES = [
    (
        "top 5 students by SGPA in semester 1",
        lambda r: (
            r.success and 0 < r.row_count <= 5,
            f"rows={r.row_count}, success={r.success}",
        ),
        "Top N ranking query",
    ),
    (
        "what is the average SGPA of all students in semester 4",
        lambda r: (r.success and r.row_count >= 1, f"rows={r.row_count}"),
        "Aggregate AVG query",
    ),
    (
        "how many students are in the CSE branch",
        lambda r: (r.success and r.row_count >= 1, f"rows={r.row_count}"),
        "COUNT query with branch filter",
    ),
    (
        "list students who got grade A+ in KCS301",
        lambda r: (r.success and r.row_count >= 0, f"rows={r.row_count}"),
        "Subject code + grade filter",
    ),
    (
        "compare average SGPA between semester 1 and semester 4 for CSE students",
        lambda r: (r.success and r.row_count >= 1, f"rows={r.row_count}"),
        "Comparison query across semesters",
    ),
    (
        "students with SGPA below 6 in semester 5",
        lambda r: (r.success, f"success={r.success}, error={r.error}"),
        "Threshold filter query",
    ),
    (
        "which student got the highest marks in Engineering Mathematics-I",
        lambda r: (r.success and r.row_count >= 1, f"rows={r.row_count}"),
        "Subject name search with MAX",
    ),
    (
        "total number of students who passed vs failed in semester 3",
        lambda r: (r.success and r.row_count >= 1, f"rows={r.row_count}"),
        "Pass/fail count comparison",
    ),
    (
        "show me the subjects where more than 10 students got grade C",
        lambda r: (r.success, f"success={r.success}, error={r.error}"),
        "GROUP BY with HAVING",
    ),
    (
        "roll number 2104920100002 results in all semesters",
        lambda r: (r.success and r.row_count >= 1, f"rows={r.row_count}"),
        "Specific student lookup across semesters",
    ),
]


async def run_tests():
    llm = OllamaClient()
    router = QueryRouter(llm)
    pipeline = SQLPipeline(llm)

    # Health check
    health = await llm.health_check()
    if health["status"] != "ok":
        print("\033[91m\u2717 Ollama not running\033[0m")
        return

    passed = 0
    failed = 0
    errors = []

    for query, validate_fn, description in TEST_CASES:
        print(f"\n{'=' * 60}")
        print(f"Test: {description}")
        print(f"Query: \"{query}\"")

        try:
            # Step 1: Route the query
            route_result = await router.route(query)
            print(f"  Route: {route_result.route} | Filters: {route_result.filters}")

            # Step 2: Run SQL pipeline (force SQL for testing)
            result = await pipeline.run(query, route_result)

            # Step 3: Print details
            if result.success:
                print(f"  SQL: {result.sql}")
                print(f"  Rows: {result.row_count} | Time: {result.execution_time_ms:.0f}ms")
                if result.rows:
                    for row in result.rows[:2]:
                        print(f"  Sample: {row}")
                print(f"  Explanation: {result.explanation}")
            else:
                print(f"  \033[91mERROR: {result.error}\033[0m")
                if result.sql:
                    print(f"  Generated SQL: {result.sql}")

            # Step 4: Validate
            ok, reason = validate_fn(result)
            if ok:
                passed += 1
                print(f"  \033[92m\u2713 PASSED\033[0m")
            else:
                failed += 1
                errors.append((query, reason))
                print(f"  \033[91m\u2717 FAILED: {reason}\033[0m")

        except Exception as e:
            failed += 1
            errors.append((query, str(e)))
            print(f"  \033[91m\u2717 EXCEPTION: {e}\033[0m")

    # Summary
    total = passed + failed
    pct = (passed / total * 100) if total > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"SQL Pipeline Results: {passed}/{total} passed ({pct:.0f}%)")
    print(f"{'=' * 60}")

    if errors:
        print(f"\n\033[91mFailed:\033[0m")
        for query, err in errors:
            print(f"  \u2022 \"{query}\" \u2014 {err}")

    if pct >= 70:
        print(f"\n\033[92m\u2713 SQL pipeline accuracy acceptable ({pct:.0f}% >= 70%)\033[0m")
    else:
        print(f"\n\033[91m\u2717 SQL pipeline accuracy too low ({pct:.0f}% < 70%). Prompt needs tuning.\033[0m")


if __name__ == "__main__":
    asyncio.run(run_tests())
