"""SQL Validator unit tests. No external dependencies needed.

Run: python -m tests.test_sql_validator
"""

from core.sql_pipeline import SQLValidator

TEST_CASES = [
    # (sql, should_pass, description)

    # === VALID QUERIES ===
    (
        "SELECT name, sgpa FROM students s JOIN semester_results sr ON s.roll_no = sr.roll_no LIMIT 10",
        True,
        "Basic SELECT with JOIN",
    ),
    (
        "SELECT COUNT(*) FROM students WHERE branch = 'COMPUTER SCIENCE AND ENGINEERING'",
        True,
        "COUNT with WHERE",
    ),
    (
        "SELECT s.name, AVG(sr.sgpa) FROM students s JOIN semester_results sr "
        "ON s.roll_no = sr.roll_no GROUP BY s.name ORDER BY AVG(sr.sgpa) DESC LIMIT 5",
        True,
        "Aggregate with GROUP BY",
    ),
    (
        "WITH cte AS (SELECT roll_no, AVG(sgpa) as avg_sgpa FROM semester_results "
        "GROUP BY roll_no) SELECT * FROM cte LIMIT 10",
        True,
        "CTE (WITH clause)",
    ),
    (
        "SELECT * FROM students WHERE name LIKE '%AAKASH%' LIMIT 10",
        True,
        "LIKE pattern matching",
    ),

    # === INVALID QUERIES ===
    (
        "DROP TABLE students",
        False,
        "DROP TABLE",
    ),
    (
        "DELETE FROM students WHERE roll_no = '123'",
        False,
        "DELETE",
    ),
    (
        "INSERT INTO students (roll_no, name) VALUES ('123', 'test')",
        False,
        "INSERT",
    ),
    (
        "UPDATE students SET name = 'hacked' WHERE roll_no = '123'",
        False,
        "UPDATE",
    ),
    (
        "SELECT * FROM students; DROP TABLE students",
        False,
        "Multiple statements (SQL injection)",
    ),
    (
        "SELECT * FROM students -- WHERE roll_no = '123'",
        False,
        "SQL comment injection",
    ),
    (
        "SELECT * FROM students /* bypass */ WHERE 1=1",
        False,
        "Block comment injection",
    ),
    (
        "SELECT SLEEP(10) FROM students",
        False,
        "SLEEP injection (DoS)",
    ),
    (
        "SELECT BENCHMARK(1000000, SHA1('test')) FROM students",
        False,
        "BENCHMARK injection (DoS)",
    ),
    (
        "SELECT * FROM INFORMATION_SCHEMA.TABLES",
        False,
        "Information schema access",
    ),
    (
        "SELECT * FROM students s "
        "JOIN semester_results sr ON s.roll_no = sr.roll_no "
        "JOIN subject_marks sm ON s.roll_no = sm.roll_no "
        "JOIN students s2 ON s.roll_no = s2.roll_no "
        "JOIN semester_results sr2 ON s.roll_no = sr2.roll_no LIMIT 10",
        False,
        "Too many JOINs (4 > max 3)",
    ),
    (
        "",
        False,
        "Empty query",
    ),
    (
        "TRUNCATE TABLE students",
        False,
        "TRUNCATE",
    ),
    (
        "ALTER TABLE students ADD COLUMN hacked VARCHAR(100)",
        False,
        "ALTER TABLE",
    ),
    (
        "SELECT * FROM students LIMIT 500",
        False,
        "LIMIT too high (500 > 100)",
    ),
]


def run_tests():
    passed = 0
    failed = 0

    for sql, should_pass, description in TEST_CASES:
        error = SQLValidator.validate(sql)
        actually_passed = error is None

        if actually_passed == should_pass:
            passed += 1
            status = "\033[92m\u2713\033[0m"
        else:
            failed += 1
            status = "\033[91m\u2717\033[0m"

        expected = "PASS" if should_pass else "BLOCK"
        print(f"  {status} [{expected}] {description}")
        if actually_passed != should_pass:
            actual = "PASS" if actually_passed else f"BLOCKED: {error}"
            print(f"    SQL: {sql[:80]}...")
            print(f"    Got: {actual}")

    total = passed + failed
    pct = (passed / total * 100) if total > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"Validator Results: {passed}/{total} passed ({pct:.0f}%)")

    if pct >= 95:
        print(f"\033[92m\u2713 Validator is solid ({pct:.0f}%)\033[0m")
    else:
        print(f"\033[91m\u2717 Validator needs fixes ({pct:.0f}%)\033[0m")


if __name__ == "__main__":
    run_tests()
