"""Hypothetical Document Embeddings (HyDE) for improved retrieval.

Instead of embedding the raw user query (which may have low vocabulary overlap
with stored chunks), we first generate a hypothetical answer containing the
kinds of terms and numbers that appear in actual chunks, then embed THAT.

Example:
- Query: "who's doing badly?" -> poor overlap with chunk vocab
- HyDE: "Student X has low SGPA of 5.2, scored C in Design and Analysis
  of Algorithm (34 external), grade C in Web Technology..." -> high overlap
"""
from __future__ import annotations

from core.llm_client import OllamaClient

HYDE_PROMPT = """Given this question about student academic data at KCCITM institute, write a short hypothetical answer paragraph that would be found in a student result record.

Include specific details like: SGPA values, grades (A+, A, B+, B, C), subject names and codes, internal/external marks, semester numbers, result status (PASS/CP/FAIL), and branch names.

Be specific and realistic — use plausible numbers and subject names from an engineering college.

Question: {query}

Hypothetical answer (1 paragraph, be specific):"""


class HyDEGenerator:
    """Generate hypothetical documents for improved embedding-based retrieval."""

    def __init__(self, llm: OllamaClient):
        self.llm = llm

    async def generate(self, query: str) -> str:
        """
        Generate a hypothetical answer document for the given query.

        Returns:
            Hypothetical answer text (~100-200 words)
        """
        prompt = HYDE_PROMPT.format(query=query)

        response = await self.llm.generate(
            prompt=prompt,
            temperature=0.5,
            max_tokens=200,
        )

        return response.strip()

    async def generate_and_embed(self, query: str) -> tuple[str, list[float]]:
        """
        Generate hypothetical doc AND embed it in one call.

        Returns:
            Tuple of (hypothetical_text, embedding_vector)
        """
        hyde_text = await self.generate(query)
        embedding = await self.llm.embed(hyde_text)
        return hyde_text, embedding
