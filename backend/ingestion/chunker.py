"""Text chunk generator: JSON → natural language chunks (one per student per semester).

Usage:
    cd backend
    python -m ingestion.chunker
"""

import json
import re
from pathlib import Path

from config import settings
from db.mysql_client import get_sync_connection

GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _strip_prefix(value: str) -> str:
    """Remove leading '(NN) ' code from course/branch strings."""
    return re.sub(r"^\(\d+\)\s*", "", value).strip()


def _clean_session(session: str) -> str:
    """'Session : 2021-22(REGULAR)' → '2021-22 (REGULAR)'."""
    s = session.replace("Session : ", "").replace("Session :", "").strip()
    # Add space before parenthesised qualifier if missing
    s = re.sub(r"(\d)\(", r"\1 (", s)
    return s


def _safe_int(value, default=None):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_float(value, default=None):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _format_subject(mark: dict) -> str:
    """Format a single subject for the chunk text."""
    name = mark.get("name", "").strip()
    grade = mark.get("grade", "").strip()
    internal = str(mark.get("internal", "")).strip()
    external = str(mark.get("external", "")).strip()

    grade_str = f" {grade}" if grade else ""

    if external:
        total = (_safe_int(internal) or 0) + (_safe_int(external) or 0)
        return f"{name}{grade_str} ({internal}+{external}={total})"
    elif internal:
        return f"{name}{grade_str} ({internal})"
    else:
        return f"{name}{grade_str}"


def generate_chunks() -> list[tuple[str, dict]]:
    """Generate all chunks from the university_marks table.

    Returns:
        List of (chunk_text, metadata_dict) tuples.
    """
    conn = get_sync_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT roll_no, jsontext FROM university_marks")
        rows = cur.fetchall()
    conn.close()

    chunks: list[tuple[str, dict]] = []
    student_count = 0

    for row in rows:
        try:
            data = json.loads(row["jsontext"])
        except (json.JSONDecodeError, TypeError):
            continue

        roll_no = str(data.get("rollno", row["roll_no"])).strip()
        name = str(data.get("name", "")).strip().upper()
        course = _strip_prefix(str(data.get("course", "")))
        branch = _strip_prefix(str(data.get("branch", "")))
        gender = str(data.get("gender", "")).strip()
        student_count += 1

        for sem in data.get("result", []):
            semester_str = str(sem.get("semester", "")).strip()
            total_subjects_str = str(sem.get("total_subjects", "0")).strip()
            sgpa_str = str(sem.get("SGPA", "0")).strip()

            # Skip empty semesters
            if semester_str == "" or total_subjects_str == "0" or sgpa_str == "0":
                continue

            semester = _safe_int(semester_str)
            if semester is None:
                continue

            sgpa = _safe_float(sgpa_str) or 0.0
            total_marks = _safe_int(sem.get("total_marks_obt")) or 0
            session = _clean_session(str(sem.get("session", "")))
            result_status = str(sem.get("result_status", "")).strip()

            # Separate theory and practical subjects
            marks = sem.get("marks", [])
            theory = [m for m in marks if str(m.get("type", "")).strip().lower() == "theory"]
            practical = [m for m in marks if str(m.get("type", "")).strip().lower() in ("practical", "ca")]

            # Build chunk text (title-case branch for readability)
            branch_display = branch.title()
            lines = [
                f"Student {name} (Roll: {roll_no}), {course} {branch_display}, "
                f"Semester {semester}, Session {session}. "
                f"SGPA: {sgpa}, Total Marks: {total_marks}. Result: {result_status}."
            ]

            if theory:
                theory_strs = [_format_subject(m) for m in theory]
                lines.append(f"Theory subjects: {', '.join(theory_strs)}.")

            if practical:
                prac_strs = [_format_subject(m) for m in practical]
                lines.append(f"Practical subjects: {', '.join(prac_strs)}.")

            chunk_text = "\n".join(lines)

            # Extract clean session year for metadata
            session_year = re.sub(r"\s*\(.*\)", "", session).strip()

            metadata = {
                "chunk_id": f"{roll_no}_sem{semester}",
                "roll_no": roll_no,
                "name": name,
                "branch": branch,
                "course": course,
                "semester": semester,
                "sgpa": sgpa,
                "session": session,
                "result_status": result_status,
                "gender": gender,
            }

            chunks.append((chunk_text, metadata))

    print(f"{GREEN}✓ Generated {len(chunks)} chunks for {student_count} students{RESET}")
    return chunks


def save_chunks(chunks: list[tuple[str, dict]], output_path: Path | None = None) -> None:
    """Save chunks to a JSONL file for downstream consumption."""
    if output_path is None:
        output_path = settings.db_path("data/chunks.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for text, meta in chunks:
            json.dump({"text": text, "metadata": meta}, f, ensure_ascii=False)
            f.write("\n")

    print(f"{GREEN}✓ Saved chunks to {output_path}{RESET}")


if __name__ == "__main__":
    chunks = generate_chunks()
    save_chunks(chunks)
