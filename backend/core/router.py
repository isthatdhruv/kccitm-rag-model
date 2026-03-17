"""Query classification and routing — classifies queries as SQL, RAG, or HYBRID
and extracts structured entities / filters for downstream pipelines.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from config import settings
from core.llm_client import OllamaClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Branch abbreviation map (used both in prompt and as post-parse fallback)
# ---------------------------------------------------------------------------

BRANCH_MAP: dict[str, str] = {
    "CSE": "COMPUTER SCIENCE AND ENGINEERING",
    "ECE": "ELECTRONICS AND COMMUNICATION ENGINEERING",
    "ME": "MECHANICAL ENGINEERING",
    "CE": "CIVIL ENGINEERING",
    "EE": "ELECTRICAL ENGINEERING",
    "IT": "INFORMATION TECHNOLOGY",
}

# ---------------------------------------------------------------------------
# Router system prompt (v1) — stored in prompts.db via init_prompts.py
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You are a query classifier for an academic student results database at KCCITM institute.

Given a user's question about student academic data, classify it and extract structured information.

DATABASE CONTEXT:
The system has student results including: student names, roll numbers, courses (B.TECH), branches (CSE, ECE, ME, etc.), semester numbers (1-8), SGPA scores, subject names with codes (like KCS503, KAS101T), internal/external marks, grades (A+, A, B+, B, C, etc.), result status (PASS, CP, FAIL), and session years.

ROUTE CLASSIFICATION:
- SQL: Use for numerical, aggregate, ranking, counting, comparison, or exact lookup queries. Examples: "top 10 students by SGPA", "average marks in semester 4", "how many students failed", "what was Aakash's SGPA", "compare semester 1 vs 6", "count of students with back papers".
- RAG: Use for descriptive, contextual, semantic, or qualitative queries. Examples: "tell me about student X", "students struggling in programming", "who performed well in practicals", "describe the CSE batch performance", "students who improved over time".
- HYBRID: Use for complex queries needing both numerical data AND contextual analysis. Examples: "why did the batch average drop in semester 6", "which students improved the most and why", "analyze the performance trend of CSE branch".

FILTER EXTRACTION:
Extract any mentioned filters:
- semester: integer 1-8 (look for "sem 1", "semester 4", "1st semester", "fifth semester", etc.)
- branch: full branch name (map "CSE" → "COMPUTER SCIENCE AND ENGINEERING", "ECE" → "ELECTRONICS AND COMMUNICATION ENGINEERING", "ME" → "MECHANICAL ENGINEERING", "CE" → "CIVIL ENGINEERING", "EE" → "ELECTRICAL ENGINEERING", "IT" → "INFORMATION TECHNOLOGY")
- roll_no: roll number if mentioned (format: 21049201000XX)
- name: student name if mentioned
- session: academic session if mentioned (e.g., "2021-22", "2023-24")
- subject_code: subject code if mentioned (e.g., "KCS503", "KAS101T")

ENTITY EXTRACTION:
Extract any specific entities mentioned: student names, roll numbers, subject names, subject codes.

Respond with ONLY a JSON object (no markdown, no explanation):
{
    "route": "SQL" | "RAG" | "HYBRID",
    "needs_filter": true | false,
    "entities": ["entity1", "entity2"],
    "filters": {
        "semester": null,
        "branch": null,
        "roll_no": null,
        "name": null,
        "session": null,
        "subject_code": null
    },
    "intent": "brief description of what the user wants",
    "complexity": "simple" | "moderate" | "complex",
    "confidence": 0.0
}"""

# ---------------------------------------------------------------------------
# Word → int map for semester extraction
# ---------------------------------------------------------------------------

_WORD_TO_INT: dict[str, int] = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
}


# ---------------------------------------------------------------------------
# Route result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RouteResult:
    """Result of query classification."""
    route: str                                          # "SQL" | "RAG" | "HYBRID"
    needs_filter: bool = False
    entities: list[str] = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    intent: str = ""
    complexity: str = "simple"
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class QueryRouter:
    """Classifies incoming queries and routes them to the optimal pipeline."""

    def __init__(self, llm: OllamaClient, prompt_store=None):
        self.llm = llm
        self.prompt_store = prompt_store  # Phase 9: dynamic prompt loading

        # Use OpenAI for routing when API key is available (much faster),
        # otherwise use the draft model — routing is just JSON classification.
        self._fast_llm = None
        if settings.OPENAI_API_KEY:
            from core.openai_client import OpenAIClient
            self._fast_llm = OpenAIClient()
        elif settings.OLLAMA_DRAFT_MODEL:
            self._fast_llm = OllamaClient(model=settings.OLLAMA_DRAFT_MODEL)

    async def route(self, query: str, chat_history: list[dict] | None = None) -> RouteResult:
        """Classify a query and determine the optimal processing pipeline."""
        system_prompt = await self._get_system_prompt()
        user_prompt = self._build_user_prompt(query, chat_history)
        llm = self._fast_llm or self.llm

        try:
            response = await llm.generate(
                prompt=user_prompt,
                system=system_prompt,
                temperature=0.1,
                max_tokens=300,
                format="json",
            )
            return self._parse_response(response, query)
        except Exception as e:
            logger.warning("LLM routing failed (%s), using fallback", e)
            return self._fallback_classify(query)

    async def _get_system_prompt(self) -> str:
        """Get the router system prompt (Phase 9 will load from prompt_store)."""
        return ROUTER_SYSTEM_PROMPT

    def _build_user_prompt(self, query: str, chat_history: list[dict] | None = None) -> str:
        """Build user prompt with optional recent context."""
        parts: list[str] = []
        if chat_history:
            recent = chat_history[-4:]  # last 2 exchanges (user+assistant each)
            parts.append("Recent conversation context:")
            for msg in recent:
                parts.append(f"  {msg['role']}: {msg['content'][:200]}")
            parts.append("")
        parts.append(f"Classify this query: {query}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, original_query: str) -> RouteResult:
        """Parse LLM JSON response into RouteResult with validation."""
        cleaned = self._clean_json(response)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed for router response, using fallback")
            return self._fallback_classify(original_query)

        if not isinstance(data, dict):
            return self._fallback_classify(original_query)

        route = str(data.get("route", "RAG")).upper()
        if route not in ("SQL", "RAG", "HYBRID"):
            route = "RAG"

        filters = self._clean_filters(data.get("filters") or {})
        entities = data.get("entities") or []
        if not isinstance(entities, list):
            entities = []

        complexity = str(data.get("complexity", "simple")).lower()
        if complexity not in ("simple", "moderate", "complex"):
            complexity = "simple"

        confidence = 0.0
        try:
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            pass

        return RouteResult(
            route=route,
            needs_filter=bool(data.get("needs_filter", False)),
            entities=[str(e) for e in entities],
            filters=filters,
            intent=str(data.get("intent", "")),
            complexity=complexity,
            confidence=confidence,
        )

    @staticmethod
    def _clean_json(text: str) -> str:
        """Strip markdown fences and whitespace from LLM output."""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        return text.strip()

    @staticmethod
    def _clean_filters(filters: dict) -> dict:
        """Validate and normalise extracted filters."""
        cleaned: dict = {}

        # Semester: must be int 1-8
        sem = filters.get("semester")
        if sem is not None:
            try:
                sem = int(sem)
                if 1 <= sem <= 8:
                    cleaned["semester"] = sem
            except (ValueError, TypeError):
                # Try word mapping
                word = str(sem).lower().strip()
                if word in _WORD_TO_INT:
                    cleaned["semester"] = _WORD_TO_INT[word]

        # Branch: map abbreviation → full name
        branch = filters.get("branch")
        if branch and str(branch).lower() != "null":
            branch = str(branch).strip().upper()
            cleaned["branch"] = BRANCH_MAP.get(branch, branch)

        # Roll number
        roll = filters.get("roll_no")
        if roll and str(roll).lower() != "null":
            cleaned["roll_no"] = str(roll).strip()

        # Name: uppercase
        name = filters.get("name")
        if name and str(name).lower() != "null":
            cleaned["name"] = str(name).strip().upper()

        # Session
        session = filters.get("session")
        if session and str(session).lower() != "null":
            cleaned["session"] = str(session).strip()

        # Subject code: uppercase
        code = filters.get("subject_code")
        if code and str(code).lower() != "null":
            cleaned["subject_code"] = str(code).strip().upper()

        return cleaned

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_classify(query: str) -> RouteResult:
        """Rule-based fallback when LLM output is unparseable."""
        q = query.lower()

        sql_keywords = [
            "top", "average", "avg", "count", "how many", "rank",
            "highest", "lowest", "maximum", "minimum", "compare",
            "list all", "above", "below", "between", "pass rate",
            "fail rate", "total", "sum", "sgpa",
        ]
        rag_keywords = [
            "tell me about", "describe", "struggling", "performing well",
            "improved", "weak", "strong", "overall", "performance of",
            "who is", "what about",
        ]

        sql_score = sum(1 for kw in sql_keywords if kw in q)
        rag_score = sum(1 for kw in rag_keywords if kw in q)

        # Extract semester filter from text
        filters: dict = {}
        sem_match = re.search(r'semester\s*(\d)', q)
        if sem_match:
            s = int(sem_match.group(1))
            if 1 <= s <= 8:
                filters["semester"] = s

        # Extract branch
        for abbr, full in BRANCH_MAP.items():
            if abbr.lower() in q.split():
                filters["branch"] = full
                break

        if sql_score > 0 and rag_score > 0:
            route = "HYBRID"
        elif sql_score > rag_score:
            route = "SQL"
        else:
            route = "RAG"

        return RouteResult(
            route=route,
            needs_filter=bool(filters),
            filters=filters,
            intent="fallback classification",
            confidence=0.3,
        )
