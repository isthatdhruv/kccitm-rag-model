"""Text-to-SQL pipeline: NL question → SQL generation → validation → execution → formatting.

The pipeline:
1. Build a schema-aware prompt with the user's question
2. LLM generates SQL + explanation as JSON
3. Safety layer validates the SQL (SELECT only, no injection, etc.)
4. Execute against MySQL with timeout
5. Format results as readable text + markdown table
"""

import json
import re
import time
from dataclasses import dataclass, field

import sqlparse

from config import settings
from core.llm_client import OllamaClient
from core.router import RouteResult
from db.mysql_client import execute_query

# ---------------------------------------------------------------------------
# System prompt — full schema + rules for the SQL generator
# ---------------------------------------------------------------------------

SQL_GENERATOR_SYSTEM_PROMPT = """You are a MySQL query generator for a student academic results database at KCCITM institute.

DATABASE SCHEMA:

Table: students
- roll_no VARCHAR(20) PRIMARY KEY — Student roll number (e.g., "2104920100002")
- name VARCHAR(255) — Student full name (uppercase, e.g., "AAKASH SINGH")
- course VARCHAR(100) — Degree program (e.g., "B.TECH")
- branch VARCHAR(200) — Branch/department (e.g., "COMPUTER SCIENCE AND ENGINEERING")
- enrollment VARCHAR(50) — Enrollment number
- father_name VARCHAR(255) — Father's name
- gender CHAR(1) — "M" or "F"

Table: semester_results
- id INT AUTO_INCREMENT PRIMARY KEY
- roll_no VARCHAR(20) — FK to students.roll_no
- semester INT — Semester number (1-8)
- session VARCHAR(100) — Academic session (e.g., "2021-22 (REGULAR)")
- sgpa DECIMAL(4,2) — Semester GPA (0.00 to 10.00)
- total_marks INT — Total marks obtained
- result_status VARCHAR(20) — "PASS", "CP(0)", "CP( 0)", "FAIL", etc.
- total_subjects INT — Number of subjects in that semester

Table: subject_marks
- id INT AUTO_INCREMENT PRIMARY KEY
- roll_no VARCHAR(20) — FK to students.roll_no
- semester INT — Semester number (1-8)
- subject_code VARCHAR(20) — Subject code (e.g., "KCS503", "KAS101T")
- subject_name VARCHAR(200) — Full subject name (e.g., "Database Management System")
- type VARCHAR(20) — "Theory", "Practical", or "CA"
- internal_marks INT — Internal/sessional marks (can be NULL)
- external_marks INT — External/exam marks (can be NULL)
- grade VARCHAR(5) — Letter grade ("A+", "A", "B+", "B", "C", "D", "F", or empty)
- back_paper VARCHAR(10) — Back paper status ("--" means no back paper)

COMMON RELATIONSHIPS:
- students.roll_no = semester_results.roll_no (one student has many semester results)
- students.roll_no = subject_marks.roll_no (one student has many subject marks)
- semester_results and subject_marks share (roll_no, semester) as a logical key

IMPORTANT NOTES:
- Branch names are FULL (e.g., "COMPUTER SCIENCE AND ENGINEERING", not "CSE")
- Student names are UPPERCASE
- SGPA is on a 10-point scale
- result_status can have variations: "PASS", "CP(0)", "CP( 0)", "CP(1)" — use LIKE for matching
- back_paper = "--" means NO back paper. Any other value means the student has a back paper.
- Some external_marks are NULL (for practicals like mini projects that only have internal marks)
- grade can be empty string for CA (Continuous Assessment) type subjects

RULES:
1. Generate ONLY SELECT statements. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, or TRUNCATE.
2. Always use table aliases for readability (s for students, sr for semester_results, sm for subject_marks).
3. Use JOINs when data from multiple tables is needed. Maximum 3 JOINs per query.
4. Always include a LIMIT clause (default LIMIT 50, or as appropriate for the question).
5. Use parameterized values where possible for user-supplied values.
6. For "top N" queries, use ORDER BY ... DESC LIMIT N.
7. For "pass/fail" queries: PASS = result_status = 'PASS', FAIL/compartment = result_status LIKE 'CP%' OR result_status = 'FAIL'.
8. For subject searches, use LIKE with wildcards: subject_name LIKE '%programming%'
9. For branch filtering, use the FULL branch name, not abbreviations.
10. Round averages to 2 decimal places: ROUND(AVG(sgpa), 2)

Respond with ONLY a JSON object (no markdown, no explanation outside the JSON):
{
    "sql": "SELECT ... FROM ... WHERE ... ORDER BY ... LIMIT ...",
    "params": [],
    "explanation": "Brief explanation of what this query does"
}"""


# ---------------------------------------------------------------------------
# SQLResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class SQLResult:
    """Result from the SQL pipeline."""

    success: bool
    sql: str = ""
    params: list = field(default_factory=list)
    explanation: str = ""
    rows: list[dict] = field(default_factory=list)
    row_count: int = 0
    formatted_text: str = ""
    formatted_table: str = ""
    error: str = ""
    execution_time_ms: float = 0


# ---------------------------------------------------------------------------
# SQL Safety Validator
# ---------------------------------------------------------------------------


class SQLValidator:
    """Validates generated SQL queries for safety before execution.

    All methods are static — no state needed.
    """

    FORBIDDEN_KEYWORDS = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE",
        "TRUNCATE", "REPLACE", "GRANT", "REVOKE", "EXEC", "EXECUTE",
        "CALL", "LOAD", "INTO OUTFILE", "INTO DUMPFILE", "INFORMATION_SCHEMA",
        "SLEEP", "BENCHMARK", "WAITFOR",
    ]

    ALLOWED_TABLES = ["students", "semester_results", "subject_marks"]

    MAX_JOINS = 3
    MAX_SUBQUERIES = 2
    MAX_RESULT_LIMIT = 100

    @staticmethod
    def validate(sql: str) -> str | None:
        """Validate SQL query. Returns None if safe, error message if unsafe."""
        if not sql or not sql.strip():
            return "Empty SQL query"

        sql_upper = sql.upper().strip()

        # Must start with SELECT (or WITH for CTEs)
        if not sql_upper.startswith("SELECT") and not sql_upper.startswith("WITH"):
            return f"Only SELECT statements allowed. Got: {sql_upper[:20]}..."

        # Forbidden keywords (word-boundary matching to avoid false positives)
        for keyword in SQLValidator.FORBIDDEN_KEYWORDS:
            pattern = r"\b" + keyword.replace(" ", r"\s+") + r"\b"
            if re.search(pattern, sql_upper):
                return f"Forbidden keyword detected: {keyword}"

        # Multiple statements
        statements = sqlparse.split(sql)
        if len(statements) > 1:
            return "Multiple SQL statements not allowed"

        # SQL comments
        if "--" in sql or "/*" in sql:
            return "SQL comments not allowed"

        # JOIN count
        join_count = len(re.findall(r"\bJOIN\b", sql_upper))
        if join_count > SQLValidator.MAX_JOINS:
            return f"Too many JOINs: {join_count} (max {SQLValidator.MAX_JOINS})"

        # Subquery count
        subquery_count = sql_upper.count("SELECT") - 1
        if subquery_count > SQLValidator.MAX_SUBQUERIES:
            return f"Too many subqueries: {subquery_count} (max {SQLValidator.MAX_SUBQUERIES})"

        # LIMIT value
        limit_match = re.search(r"LIMIT\s+(\d+)", sql_upper)
        if limit_match:
            limit_val = int(limit_match.group(1))
            if limit_val > SQLValidator.MAX_RESULT_LIMIT:
                return f"LIMIT too high: {limit_val} (max {SQLValidator.MAX_RESULT_LIMIT})"

        return None

    @staticmethod
    def enforce_limit(sql: str, max_limit: int = 100) -> str:
        """Add or reduce LIMIT clause to enforce max_limit."""
        limit_match = re.search(r"LIMIT\s+(\d+)", sql, flags=re.IGNORECASE)

        if limit_match:
            current_limit = int(limit_match.group(1))
            if current_limit > max_limit:
                sql = re.sub(
                    r"LIMIT\s+\d+", f"LIMIT {max_limit}", sql, flags=re.IGNORECASE
                )
        else:
            sql = sql.rstrip(";").strip() + f" LIMIT {max_limit}"

        return sql


# ---------------------------------------------------------------------------
# SQL Pipeline
# ---------------------------------------------------------------------------


class SQLPipeline:
    """Converts natural language queries to SQL, validates, executes, and formats results."""

    def __init__(self, llm: OllamaClient):
        self.llm = llm
        self.max_rows = 100
        self.query_timeout = 5

        # Use OpenAI for SQL generation when API key is available (much faster)
        self._fast_llm = None
        if settings.OPENAI_API_KEY:
            from core.openai_client import OpenAIClient
            self._fast_llm = OpenAIClient()

    async def run(self, query: str, route_result: RouteResult) -> SQLResult:
        """Full pipeline: NL question -> SQL -> validate -> execute -> format."""
        # Step 1: Generate SQL
        generated = await self._generate_sql(query, route_result)
        if not generated.success:
            return generated

        # Step 2: Validate SQL
        validation_error = self._validate_sql(generated.sql)
        if validation_error:
            return SQLResult(
                success=False,
                sql=generated.sql,
                error=f"SQL validation failed: {validation_error}",
            )

        # Step 3: Execute
        result = await self._execute_sql(generated)
        if not result.success:
            return result

        # Step 4: Format results
        result.formatted_text = self._format_as_text(
            result.rows, result.sql, generated.explanation
        )
        result.formatted_table = self._format_as_markdown_table(result.rows)

        return result

    # ------------------------------------------------------------------
    # SQL generation
    # ------------------------------------------------------------------

    async def _generate_sql(self, query: str, route_result: RouteResult) -> SQLResult:
        """Use LLM to generate SQL from natural language.

        Uses OpenAI (fast) when available, falls back to Ollama (local).
        """
        system_prompt = SQL_GENERATOR_SYSTEM_PROMPT
        user_prompt = self._build_user_prompt(query, route_result)
        llm = self._fast_llm or self.llm

        try:
            response = await llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.1,
                max_tokens=500,
                format="json",
            )
            return self._parse_sql_response(response)
        except Exception as e:
            return SQLResult(success=False, error=f"SQL generation failed: {e}")

    def _build_user_prompt(self, query: str, route_result: RouteResult) -> str:
        """Build the user prompt with question and router context."""
        parts = [f"Question: {query}"]

        if route_result.intent:
            parts.append(f"Intent: {route_result.intent}")

        if route_result.filters:
            filter_strs = [
                f"{k}={v}" for k, v in route_result.filters.items() if v is not None
            ]
            if filter_strs:
                parts.append(f"Filters: {', '.join(filter_strs)}")

        if route_result.entities:
            parts.append(f"Entities: {', '.join(route_result.entities)}")

        return "\n".join(parts)

    def _parse_sql_response(self, response: str) -> SQLResult:
        """Parse LLM's JSON response into SQLResult.

        Handles: clean JSON, markdown fences, missing keys, fallback regex extraction.
        """
        text = response.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
            sql = data.get("sql", "").strip()
            if not sql:
                return SQLResult(success=False, error="LLM returned empty SQL")

            params = data.get("params", [])

            # Convert ? placeholders to %s for MySQL (LLM sometimes uses SQLite-style)
            if "?" in sql and params:
                sql = sql.replace("?", "%s")

            return SQLResult(
                success=True,
                sql=sql,
                params=params,
                explanation=data.get("explanation", ""),
            )
        except json.JSONDecodeError:
            # Fallback: extract SQL from raw text
            match = re.search(
                r"(SELECT\s+.+?)(?:;|\Z)", text, re.IGNORECASE | re.DOTALL
            )
            if match:
                return SQLResult(
                    success=True,
                    sql=match.group(1).strip(),
                    explanation="(extracted from raw response)",
                )
            return SQLResult(
                success=False,
                error=f"Cannot parse LLM response: {text[:200]}",
            )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_sql(self, sql: str) -> str | None:
        """Validate generated SQL for safety."""
        return SQLValidator.validate(sql)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def _execute_sql(self, generated: SQLResult) -> SQLResult:
        """Execute the validated SQL against MySQL."""
        sql = SQLValidator.enforce_limit(generated.sql, self.max_rows)
        params = tuple(generated.params) if generated.params else None

        try:
            start = time.time()
            rows = await execute_query(sql, params)
            elapsed = (time.time() - start) * 1000

            return SQLResult(
                success=True,
                sql=generated.sql,
                params=generated.params,
                explanation=generated.explanation,
                rows=rows,
                row_count=len(rows),
                execution_time_ms=elapsed,
            )
        except Exception as e:
            return SQLResult(
                success=False,
                sql=generated.sql,
                params=generated.params,
                explanation=generated.explanation,
                error=f"SQL execution error: {e}",
            )

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _format_as_text(self, rows: list[dict], sql: str, explanation: str) -> str:
        """Format SQL results as human-readable text for LLM context."""
        if not rows:
            return "Query returned 0 rows. No matching data found."

        parts = [f"Query returned {len(rows)} row{'s' if len(rows) != 1 else ''}."]

        if explanation:
            parts.append(f"Explanation: {explanation}")

        parts.append("\nResults:")

        display_rows = rows[:20]
        for i, row in enumerate(display_rows, 1):
            values = []
            for k, v in row.items():
                col = k.replace("_", " ").title()
                if isinstance(v, float):
                    v = round(v, 2)
                values.append(f"{col}: {v}")
            parts.append(f"  {i}. {', '.join(values)}")

        if len(rows) > 20:
            parts.append(f"  ... and {len(rows) - 20} more rows")

        return "\n".join(parts)

    def _format_as_markdown_table(self, rows: list[dict]) -> str:
        """Format SQL results as a markdown table for frontend display."""
        if not rows:
            return ""

        display_rows = rows[:25]
        columns = list(display_rows[0].keys())

        # Header row
        headers = [col.replace("_", " ").title() for col in columns]
        header_line = "| " + " | ".join(headers) + " |"
        separator = "| " + " | ".join("---" for _ in columns) + " |"

        # Data rows
        data_lines = []
        for row in display_rows:
            cells = []
            for col in columns:
                val = row.get(col, "")
                if isinstance(val, float):
                    val = round(val, 2)
                val_str = str(val) if val is not None else ""
                if len(val_str) > 40:
                    val_str = val_str[:37] + "..."
                cells.append(val_str)
            data_lines.append("| " + " | ".join(cells) + " |")

        lines = [header_line, separator] + data_lines

        if len(rows) > 25:
            lines.append(f"\n*... and {len(rows) - 25} more rows*")

        return "\n".join(lines)
