"""Router classification test suite — calls actual Ollama LLM for classification.

Run: python -m tests.test_router
"""

import asyncio

from core.llm_client import OllamaClient
from core.router import QueryRouter

# (query, expected_route, expected_filters_subset, description)
TEST_CASES = [
    # === SQL ROUTE ===
    ("top 5 students by SGPA in semester 4",
     "SQL", {"semester": 4}, "Ranking query → SQL"),

    ("what is the average SGPA of CSE students",
     "SQL", {"branch": "COMPUTER SCIENCE AND ENGINEERING"}, "Aggregate with branch → SQL"),

    ("how many students failed in semester 3",
     "SQL", {"semester": 3}, "Count query → SQL"),

    ("what was Aakash Singh's SGPA in semester 1",
     "SQL", {"semester": 1}, "Specific lookup → SQL"),

    ("compare pass rates between semester 1 and semester 6",
     "SQL", {}, "Comparison → SQL"),

    ("count students with grade C in KCS503",
     "SQL", {"subject_code": "KCS503"}, "Subject-specific count → SQL"),

    ("list all students with SGPA above 9",
     "SQL", {}, "Threshold query → SQL"),

    # === RAG ROUTE ===
    ("tell me about roll number 2104920100002",
     "RAG", {"roll_no": "2104920100002"}, "Descriptive → RAG"),

    ("which students are struggling in programming subjects",
     "RAG", {}, "Semantic/qualitative → RAG"),

    ("describe the overall performance of the 2021 CSE batch",
     "RAG", {"branch": "COMPUTER SCIENCE AND ENGINEERING"}, "Descriptive analysis → RAG"),

    ("students who performed well in practicals",
     "RAG", {}, "Qualitative → RAG"),

    ("who are the weak students in semester 5",
     "RAG", {"semester": 5}, "Qualitative with filter → RAG"),

    # === HYBRID ROUTE ===
    ("why did the average SGPA drop in semester 6 compared to semester 4",
     "HYBRID", {}, "Needs data + analysis → HYBRID"),

    ("which CSE students improved the most from semester 1 to semester 4 and why",
     "HYBRID", {"branch": "COMPUTER SCIENCE AND ENGINEERING"}, "Trend + explanation → HYBRID"),

    # === EDGE CASES ===
    ("tell me about subject KCS503",
     "RAG", {"subject_code": "KCS503"}, "Subject info request → RAG"),

    ("hello",
     "RAG", {}, "Greeting → RAG (fallback)"),

    ("what about semester 3",
     "SQL", {"semester": 3}, "Ambiguous follow-up → best guess"),

    # === BRANCH ABBREVIATION MAPPING ===
    ("top CSE students",
     "SQL", {"branch": "COMPUTER SCIENCE AND ENGINEERING"}, "CSE abbreviation mapped"),

    ("ECE semester 4 results",
     "SQL", {"branch": "ELECTRONICS AND COMMUNICATION ENGINEERING", "semester": 4}, "ECE mapped"),
]


async def run_tests() -> None:
    llm = OllamaClient()
    router = QueryRouter(llm)

    health = await llm.health_check()
    if health["status"] != "ok":
        print(f"\033[91m\u2717 Ollama not running: {health.get('message')}\033[0m")
        return

    passed = 0
    failed = 0
    errors: list[tuple[str, str]] = []

    for query, expected_route, expected_filters, description in TEST_CASES:
        try:
            result = await router.route(query)

            route_ok = result.route == expected_route
            filters_ok = all(
                result.filters.get(k) == v
                for k, v in expected_filters.items()
                if v is not None
            )

            if route_ok and filters_ok:
                passed += 1
                print(f"\033[92m\u2713 {description}\033[0m")
                print(f"  Query: \"{query}\"")
                print(f"  Route: {result.route} | Filters: {result.filters} | Intent: {result.intent}")
            else:
                failed += 1
                err_parts = []
                if not route_ok:
                    err_parts.append(f"route: expected {expected_route}, got {result.route}")
                if not filters_ok:
                    err_parts.append(f"filters: expected {expected_filters}, got {result.filters}")
                error_msg = " | ".join(err_parts)
                errors.append((query, error_msg))
                print(f"\033[91m\u2717 {description}\033[0m")
                print(f"  Query: \"{query}\"")
                print(f"  {error_msg}")
                print(f"  Full result: route={result.route}, filters={result.filters}, intent={result.intent}")

        except Exception as e:
            failed += 1
            errors.append((query, str(e)))
            print(f"\033[91m\u2717 {description} — ERROR: {e}\033[0m")

        print()

    total = passed + failed
    pct = (passed / total * 100) if total > 0 else 0
    print(f"\n{'=' * 60}")
    print(f"Router Test Results: {passed}/{total} passed ({pct:.0f}%)")
    print(f"{'=' * 60}")

    if errors:
        print(f"\n\033[91mFailed cases:\033[0m")
        for query, err in errors:
            print(f"  \u2022 \"{query}\" — {err}")

    if pct >= 75:
        print(f"\n\033[92m\u2713 Router accuracy is acceptable ({pct:.0f}% >= 75% threshold)\033[0m")
    else:
        print(f"\n\033[91m\u2717 Router accuracy too low ({pct:.0f}% < 75% threshold). Prompt needs tuning.\033[0m")


if __name__ == "__main__":
    asyncio.run(run_tests())
