"""Initialize the prompts database with v1 system prompts.

Run: python -m ingestion.init_prompts
"""

import asyncio
import uuid

from config import settings
from core.rag_pipeline import RESPONSE_GENERATOR_PROMPT
from core.router import ROUTER_SYSTEM_PROMPT
from core.sql_pipeline import SQL_GENERATOR_SYSTEM_PROMPT
from db.sqlite_client import execute, fetch_one

INITIAL_PROMPTS = [
    {
        "prompt_name": "router",
        "section_name": "system",
        "content": ROUTER_SYSTEM_PROMPT,
    },
    {
        "prompt_name": "sql_generator",
        "section_name": "system",
        "content": SQL_GENERATOR_SYSTEM_PROMPT,
    },
    {
        "prompt_name": "response_generator",
        "section_name": "system",
        "content": RESPONSE_GENERATOR_PROMPT,
    },
    {
        "prompt_name": "response_generator",
        "section_name": "persona",
        "content": RESPONSE_GENERATOR_PROMPT.split("RULES:")[1].split("CONTEXT TYPE:")[0].strip()
        if "RULES:" in RESPONSE_GENERATOR_PROMPT
        else "PLACEHOLDER",
    },
    {
        "prompt_name": "hyde",
        "section_name": "system",
        "content": "PLACEHOLDER — will be implemented in Phase 5",
    },
    {
        "prompt_name": "multi_query",
        "section_name": "system",
        "content": "PLACEHOLDER — will be implemented in Phase 5",
    },
    {
        "prompt_name": "compressor",
        "section_name": "system",
        "content": "PLACEHOLDER — will be implemented in Phase 5",
    },
]


async def init_prompts() -> None:
    """Store initial prompts in prompts.db (idempotent)."""
    db = settings.db_path(settings.PROMPTS_DB)

    for prompt in INITIAL_PROMPTS:
        existing = await fetch_one(
            db,
            "SELECT id FROM prompt_templates WHERE prompt_name = ? AND section_name = ? AND is_active = 1",
            (prompt["prompt_name"], prompt["section_name"]),
        )
        if existing:
            print(f"  Prompt '{prompt['prompt_name']}/{prompt['section_name']}' already exists, skipping")
            continue

        await execute(
            db,
            """INSERT INTO prompt_templates (id, prompt_name, section_name, content, version, is_active)
               VALUES (?, ?, ?, ?, 1, 1)""",
            (str(uuid.uuid4()), prompt["prompt_name"], prompt["section_name"], prompt["content"]),
        )
        status = "\u2713" if "PLACEHOLDER" not in prompt["content"] else "\u25cb"
        print(f"  {status} Stored '{prompt['prompt_name']}/{prompt['section_name']}' v1")

    print(f"\n\033[92m\u2713 All initial prompts stored in {db}\033[0m")


async def update_sql_prompt() -> None:
    """Replace the sql_generator placeholder (if it exists) with the real prompt."""
    db = settings.db_path(settings.PROMPTS_DB)

    existing = await fetch_one(
        db,
        "SELECT id, content FROM prompt_templates WHERE prompt_name = ? AND section_name = ? AND is_active = 1",
        ("sql_generator", "system"),
    )
    if not existing:
        print("  sql_generator/system not found — run init_prompts first")
        return

    if "PLACEHOLDER" in existing["content"]:
        await execute(
            db,
            "UPDATE prompt_templates SET content = ?, version = 1 WHERE id = ?",
            (SQL_GENERATOR_SYSTEM_PROMPT, existing["id"]),
        )
        print("  \033[92m\u2713 Updated sql_generator/system prompt (replaced placeholder)\033[0m")
    else:
        print("  sql_generator/system already has real content, skipping")


async def update_response_prompt() -> None:
    """Replace the response_generator placeholder (if it exists) with the real prompt."""
    db = settings.db_path(settings.PROMPTS_DB)

    for section, content in [
        ("system", RESPONSE_GENERATOR_PROMPT),
        (
            "persona",
            RESPONSE_GENERATOR_PROMPT.split("RULES:")[1].split("CONTEXT TYPE:")[0].strip()
            if "RULES:" in RESPONSE_GENERATOR_PROMPT
            else "",
        ),
    ]:
        existing = await fetch_one(
            db,
            "SELECT id, content FROM prompt_templates WHERE prompt_name = ? AND section_name = ? AND is_active = 1",
            ("response_generator", section),
        )
        if not existing:
            print(f"  response_generator/{section} not found — run init_prompts first")
            continue

        if "PLACEHOLDER" in existing["content"]:
            await execute(
                db,
                "UPDATE prompt_templates SET content = ?, version = 1 WHERE id = ?",
                (content, existing["id"]),
            )
            print(f"  \033[92m✓ Updated response_generator/{section} prompt (replaced placeholder)\033[0m")
        else:
            print(f"  response_generator/{section} already has real content, skipping")


if __name__ == "__main__":
    asyncio.run(init_prompts())
    asyncio.run(update_sql_prompt())
    asyncio.run(update_response_prompt())
