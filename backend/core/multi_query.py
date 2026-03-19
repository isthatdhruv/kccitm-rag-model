"""Multi-query expansion with Reciprocal Rank Fusion (RRF).

Expands a single query into multiple variants for broader retrieval.
Each variant is searched independently, and results are merged using RRF,
which boosts chunks that appear across multiple query interpretations.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict

from core.llm_client import OllamaClient

MULTI_QUERY_PROMPT = """Rewrite the following question about student academic data in 3 different ways. Each rewrite should emphasize a different aspect of the question.

Rules:
- Each variant should be a complete, self-contained question
- Vary the vocabulary: use synonyms, rephrase, change structure
- If the original mentions abbreviations (CSE, ECE), expand them in at least one variant
- If the original is vague, make variants more specific
- Keep each variant under 30 words

Original question: {query}

Return ONLY a JSON array of exactly 3 strings. No explanation, no markdown:
["variant 1", "variant 2", "variant 3"]"""


class MultiQueryExpander:
    """Expand a single query into multiple variants for broader retrieval."""

    def __init__(self, llm: OllamaClient):
        self.llm = llm

    async def expand(self, query: str) -> list[str]:
        """
        Generate 3 query variants.

        Returns:
            List of up to 3 variant queries (does NOT include the original)
        """
        prompt = MULTI_QUERY_PROMPT.format(query=query)

        try:
            response = await self.llm.generate(
                prompt=prompt,
                temperature=0.4,
                max_tokens=250,
                format="json",
            )

            return self._parse_variants(response)

        except Exception as e:
            print(f"Multi-query expansion failed: {e}")
            return []

    def _parse_variants(self, response: str) -> list[str]:
        """Parse LLM response into list of variant strings."""
        text = response.strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                variants = [str(v).strip() for v in parsed if v and str(v).strip()]
                return variants[:3]
            if isinstance(parsed, dict) and "variants" in parsed:
                return [str(v).strip() for v in parsed["variants"]][:3]
        except json.JSONDecodeError:
            pass

        # Fallback: extract quoted strings
        matches = re.findall(r'"([^"]{10,})"', text)
        return matches[:3]

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: list[list[dict]],
        k: int = 60,
        id_field: str = "chunk_id",
    ) -> list[dict]:
        """
        Merge multiple ranked result lists using Reciprocal Rank Fusion.

        RRF score for document d = sum over all lists of: 1 / (k + rank(d))
        Chunks that appear in multiple lists get boosted scores.

        Args:
            result_lists: List of result lists, each [{chunk_id, text, metadata, score}, ...]
            k: RRF parameter (higher = less aggressive ranking). Default 60.
            id_field: Field used as unique ID for deduplication

        Returns:
            Merged and sorted list of unique results with RRF scores
        """
        scores: dict[str, float] = defaultdict(float)
        docs: dict[str, dict] = {}

        for result_list in result_lists:
            for rank, doc in enumerate(result_list):
                doc_id = doc.get(id_field)
                if not doc_id:
                    continue
                scores[doc_id] += 1.0 / (k + rank + 1)
                if doc_id not in docs or doc.get("score", 0) > docs[doc_id].get("score", 0):
                    docs[doc_id] = doc

        sorted_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

        merged = []
        for doc_id in sorted_ids:
            doc = docs[doc_id].copy()
            doc["rrf_score"] = scores[doc_id]
            merged.append(doc)

        return merged
