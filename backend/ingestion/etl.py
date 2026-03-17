"""ETL pipeline: read university_marks JSON → normalised MySQL tables.

Usage:
    cd backend
    python -m ingestion.etl
"""

import json
import re
import time

import pymysql

from config import settings
from db.mysql_client import get_sync_connection

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

BATCH_SIZE = 500


def _strip_prefix(value: str) -> str:
    """Remove leading '(NN) ' code from course/branch strings."""
    return re.sub(r"^\(\d+\)\s*", "", value).strip()


def _safe_int(value, default=None):
    """Convert to int, returning default on failure."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value, default=None):
    """Convert to float, returning default on failure."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Table DDL
# ---------------------------------------------------------------------------

CREATE_STUDENTS = """
CREATE TABLE IF NOT EXISTS students (
    roll_no VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255),
    course VARCHAR(100),
    branch VARCHAR(300),
    enrollment VARCHAR(100),
    father_name VARCHAR(255),
    gender VARCHAR(5)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

CREATE_SEMESTER_RESULTS = """
CREATE TABLE IF NOT EXISTS semester_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    roll_no VARCHAR(50) NOT NULL,
    semester INT NOT NULL,
    session VARCHAR(100),
    sgpa DECIMAL(4,2),
    total_marks INT,
    result_status VARCHAR(50),
    total_subjects INT,
    UNIQUE KEY uq_roll_sem (roll_no, semester),
    FOREIGN KEY (roll_no) REFERENCES students(roll_no) ON DELETE CASCADE
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

CREATE_SUBJECT_MARKS = """
CREATE TABLE IF NOT EXISTS subject_marks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    roll_no VARCHAR(50) NOT NULL,
    semester INT NOT NULL,
    subject_code VARCHAR(50),
    subject_name VARCHAR(255),
    type VARCHAR(50),
    internal_marks INT,
    external_marks INT,
    grade VARCHAR(10),
    back_paper VARCHAR(10),
    FOREIGN KEY (roll_no) REFERENCES students(roll_no) ON DELETE CASCADE
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""

INSERT_STUDENT = """
INSERT INTO students (roll_no, name, course, branch, enrollment, father_name, gender)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

INSERT_SEMESTER = """
INSERT INTO semester_results (roll_no, semester, session, sgpa, total_marks, result_status, total_subjects)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
    session = VALUES(session), sgpa = VALUES(sgpa), total_marks = VALUES(total_marks),
    result_status = VALUES(result_status), total_subjects = VALUES(total_subjects)
"""

INSERT_SUBJECT = """
INSERT IGNORE INTO subject_marks (roll_no, semester, subject_code, subject_name, type, internal_marks, external_marks, grade, back_paper)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _is_empty_semester(sem: dict) -> bool:
    """Return True if the semester entry should be skipped."""
    semester_val = str(sem.get("semester", "")).strip()
    total_subj = str(sem.get("total_subjects", "0")).strip()
    sgpa = str(sem.get("SGPA", "0")).strip()
    return semester_val == "" or total_subj == "0" or sgpa == "0"


def run_etl() -> None:
    """Execute the full ETL pipeline."""
    conn = get_sync_connection()
    start = time.time()

    print(f"{YELLOW}Starting ETL pipeline...{RESET}")

    # Drop and recreate tables (idempotent)
    with conn.cursor() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS = 0")
        cur.execute("DROP TABLE IF EXISTS subject_marks")
        cur.execute("DROP TABLE IF EXISTS semester_results")
        cur.execute("DROP TABLE IF EXISTS students")
        cur.execute("SET FOREIGN_KEY_CHECKS = 1")
        cur.execute(CREATE_STUDENTS)
        cur.execute(CREATE_SEMESTER_RESULTS)
        cur.execute(CREATE_SUBJECT_MARKS)
    print(f"{GREEN}✓ Tables created{RESET}")

    # Read source data
    with conn.cursor() as cur:
        cur.execute("SELECT roll_no, jsontext FROM university_marks")
        rows = cur.fetchall()
    total = len(rows)
    print(f"  Found {total} records in university_marks")

    student_batch: list[tuple] = []
    semester_batch: list[tuple] = []
    subject_batch: list[tuple] = []

    students_count = 0
    semesters_count = 0
    subjects_count = 0
    skipped = 0

    for idx, row in enumerate(rows, 1):
        try:
            data = json.loads(row["jsontext"])
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"{RED}  ✗ Bad JSON for roll_no={row['roll_no']}: {exc}{RESET}")
            skipped += 1
            continue

        roll_no = str(data.get("rollno", row["roll_no"])).strip()
        name = str(data.get("name", "")).strip().upper()
        course = _strip_prefix(str(data.get("course", "")))
        branch = _strip_prefix(str(data.get("branch", "")))
        enrollment = str(data.get("enrollment", "")).strip()
        father_name = str(data.get("fname", "")).strip()
        gender = str(data.get("gender", "")).strip()

        student_batch.append((roll_no, name, course, branch, enrollment, father_name, gender))
        students_count += 1

        for sem in data.get("result", []):
            if _is_empty_semester(sem):
                continue

            semester = _safe_int(sem.get("semester"))
            if semester is None:
                continue

            session = str(sem.get("session", "")).strip()
            sgpa = _safe_float(sem.get("SGPA"))
            total_marks = _safe_int(sem.get("total_marks_obt"))
            result_status = str(sem.get("result_status", "")).strip()
            total_subjects = _safe_int(sem.get("total_subjects"))

            semester_batch.append((roll_no, semester, session, sgpa, total_marks, result_status, total_subjects))
            semesters_count += 1

            for mark in sem.get("marks", []):
                subject_code = str(mark.get("code", "")).strip()
                subject_name = str(mark.get("name", "")).strip()
                subj_type = str(mark.get("type", "")).strip()
                internal = _safe_int(mark.get("internal"))
                external = _safe_int(mark.get("external")) if str(mark.get("external", "")).strip() != "" else None
                grade = str(mark.get("grade", "")).strip()
                back_paper = str(mark.get("back_paper", "")).strip()

                subject_batch.append((roll_no, semester, subject_code, subject_name, subj_type, internal, external, grade, back_paper))
                subjects_count += 1

        # Flush batches periodically
        if len(student_batch) >= BATCH_SIZE:
            _flush_batches(conn, student_batch, semester_batch, subject_batch)
            student_batch.clear()
            semester_batch.clear()
            subject_batch.clear()

        if idx % 500 == 0 or idx == total:
            print(f"  Processing student {idx}/{total}...")

    # Flush remaining
    if student_batch:
        _flush_batches(conn, student_batch, semester_batch, subject_batch)

    conn.close()

    elapsed = time.time() - start
    print(f"\n{GREEN}✓ ETL complete in {elapsed:.1f}s.{RESET}")
    print(f"  {students_count} students, {semesters_count} semester records, {subjects_count} subject marks.")
    if skipped:
        print(f"  {YELLOW}{skipped} records skipped (bad JSON){RESET}")


def _flush_batches(
    conn: pymysql.Connection,
    students: list[tuple],
    semesters: list[tuple],
    subjects: list[tuple],
) -> None:
    """Insert accumulated batches into MySQL."""
    with conn.cursor() as cur:
        if students:
            cur.executemany(INSERT_STUDENT, students)
        if semesters:
            cur.executemany(INSERT_SEMESTER, semesters)
        if subjects:
            cur.executemany(INSERT_SUBJECT, subjects)


if __name__ == "__main__":
    run_etl()
