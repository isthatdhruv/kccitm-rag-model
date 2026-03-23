"""Contextual compression — strip irrelevant content from chunks before LLM generation.

Instead of sending 10 full chunks (each with 10+ subjects, ~400 tokens),
the compressor extracts only the parts relevant to the query,
typically reducing token usage by 40-60%.

The compression happens in a SINGLE LLM call — all chunks are batched together
with numbered entries, and the LLM returns numbered compressed versions.
"""
from __future__ import annotations

import re

from core.llm_client import OllamaClient

COMPRESSION_PROMPT = """You are a data extraction assistant. Given a user's question and numbered student records, extract ONLY the information relevant to the question from each record.

Rules:
- Keep specific data points: SGPA values, marks (internal+external), grades, subject names
- Remove subjects and details that are NOT relevant to the question
- If a record has nothing relevant, write "IRRELEVANT" for that number
- Keep the student's name, roll number, branch, and semester in each extract
- Be concise but preserve all relevant numbers
- Return numbered extracts matching the input numbers

Question: {query}

Records:
{numbered_records}

Relevant extracts (numbered, one per line):"""


class ContextualCompressor:
    """Compress retrieved chunks by extracting only query-relevant content."""

    def __init__(self, llm: OllamaClient):
        self.llm = llm

    async def compress(
        self,
        query: str,
        chunks: list[dict],
        text_field: str = "text",
    ) -> list[dict]:
        """
        Compress chunks by extracting only query-relevant content.

        Chunks marked IRRELEVANT by the LLM are removed entirely.
        On failure, returns original chunks (no data loss).
        """
        if not chunks:
            return []

        # Build numbered records for batch processing
        numbered = []
        for i, chunk in enumerate(chunks):
            text = chunk.get(text_field, "")
            numbered.append(f"{i + 1}. {text}")

        numbered_text = "\n\n".join(numbered)

        prompt = COMPRESSION_PROMPT.format(
            query=query,
            numbered_records=numbered_text,
        )

        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1500,
            )

            return self._parse_compressed(response, chunks, text_field)

        except Exception as e:
            print(f"Compression failed, using original chunks: {e}")
            return chunks

    def _parse_compressed(
        self,
        response: str,
        original_chunks: list[dict],
        text_field: str,
    ) -> list[dict]:
        """Parse numbered compressed output and map back to original chunks."""
        result = []
        lines = response.strip().split("\n")

        compressed_map: dict[int, str] = {}
        current_num: int | None = None
        current_text: list[str] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(r"^(\d+)[.\):\s]+(.+)", line)
            if match:
                if current_num is not None:
                    compressed_map[current_num] = "\n".join(current_text).strip()

                current_num = int(match.group(1))
                current_text = [match.group(2)]
            elif current_num is not None:
                current_text.append(line)

        if current_num is not None:
            compressed_map[current_num] = "\n".join(current_text).strip()

        for i, chunk in enumerate(original_chunks):
            num = i + 1
            compressed_text = compressed_map.get(num, "")

            if "IRRELEVANT" in compressed_text.upper():
                continue

            chunk_copy = chunk.copy()
            if compressed_text and len(compressed_text) > 10:
                chunk_copy[text_field] = compressed_text
                chunk_copy["compressed"] = True
            else:
                chunk_copy["compressed"] = False

            result.append(chunk_copy)

        # Safety net: if compression removed ALL chunks, return originals
        if not result:
            return original_chunks

        return result

    def estimate_savings(
        self,
        original_chunks: list[dict],
        compressed_chunks: list[dict],
        text_field: str = "text",
    ) -> dict:
        """Calculate token savings from compression."""
        from core.context_builder import ContextBuilder

        cb = ContextBuilder()

        orig_tokens = sum(cb.count_tokens(c.get(text_field, "")) for c in original_chunks)
        comp_tokens = sum(cb.count_tokens(c.get(text_field, "")) for c in compressed_chunks)
        saved = orig_tokens - comp_tokens
        pct = (saved / orig_tokens * 100) if orig_tokens > 0 else 0

        return {
            "original_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "saved_tokens": saved,
            "savings_percent": round(pct, 1),
            "chunks_removed": len(original_chunks) - len(compressed_chunks),
        }
