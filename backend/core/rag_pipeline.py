"""RAG pipeline: embed query → Milvus hybrid search → assemble context → generate response.

This is the Phase 4 (basic) version. Phase 5 adds HyDE, multi-query expansion,
cross-encoder re-ranking, and contextual compression on top of this.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import settings
from core.llm_client import OllamaClient
from core.router import RouteResult
from db.milvus_client import MilvusSearchClient

# ---------------------------------------------------------------------------
# Response generator system prompt
# ---------------------------------------------------------------------------

RESPONSE_GENERATOR_PROMPT = """You are KCCITM AI Assistant, an expert academic data analyst for KCCITM institute. You help faculty and administrators understand student performance data.

CAPABILITIES:
- Analyze student results across semesters, subjects, and branches
- Compare performance trends and identify patterns
- Provide statistical insights (averages, rankings, distributions)
- Answer specific questions about individual students or groups

RULES:
1. Always cite specific data points from the provided context (SGPA values, marks, grades, subject names).
2. When showing rankings or lists, include the actual numbers.
3. If the provided data is insufficient to answer the question, say so honestly. Never fabricate data.
4. Format comparisons as markdown tables when there are 3+ items to compare.
5. Be concise but thorough — cover the key points without unnecessary filler.
6. When you notice interesting patterns beyond the direct question, briefly mention them.
7. At the end of your response, suggest 2-3 related follow-up questions the user might want to ask.
8. Reference students by name and roll number when discussing individual performance.
9. When discussing grades: A+ is excellent, A is very good, B+ is good, B is above average, C is average, D is below average.
10. SGPA above 8.0 is strong, 6.0-8.0 is moderate, below 6.0 needs attention.

CONTEXT TYPE: RAG (Retrieved student data chunks)
The data below was retrieved from the student results database based on relevance to the question. Use it to form your answer."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """Result from the RAG pipeline."""

    success: bool
    chunks: list[dict] = field(default_factory=list)
    chunk_count: int = 0
    context_text: str = ""
    response: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# RAG Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    Basic RAG pipeline: embed query → Milvus hybrid search → assemble context → generate response.
    """

    def __init__(self, llm: OllamaClient, milvus: MilvusSearchClient):
        self.llm = llm
        self.milvus = milvus

    async def run(
        self,
        query: str,
        route_result: RouteResult,
        chat_history: list[dict] | None = None,
    ) -> RAGResult:
        """
        Full RAG pipeline: retrieve relevant chunks and generate a response.

        Args:
            query: User's natural language question
            route_result: RouteResult from router (has filters, entities, intent)
            chat_history: Optional conversation history for context

        Returns:
            RAGResult with retrieved chunks, assembled context, and generated response
        """
        try:
            # Step 1: Embed the query
            query_embedding = await self.llm.embed(query)

            # Step 2: Retrieve from Milvus (hybrid: dense + BM25 + optional filters)
            chunks = self._retrieve(query, query_embedding, route_result)

            if not chunks:
                return RAGResult(
                    success=True,
                    chunks=[],
                    chunk_count=0,
                    context_text="",
                    response="I couldn't find any relevant student data for your query. Could you rephrase or provide more details?",
                )

            # Step 3: Assemble context (chunks + chat history → formatted text)
            context_text = self._assemble_context(chunks, chat_history)

            # Step 4: Generate response using LLM
            response = await self._generate_response(query, context_text, chat_history)

            return RAGResult(
                success=True,
                chunks=chunks,
                chunk_count=len(chunks),
                context_text=context_text,
                response=response,
            )

        except Exception as e:
            return RAGResult(success=False, error=f"RAG pipeline error: {str(e)}")

    async def retrieve_only(
        self,
        query: str,
        route_result: RouteResult,
    ) -> list[dict]:
        """
        Retrieve chunks without generating a response.
        Useful for the HYBRID route where SQL pipeline also runs.
        """
        query_embedding = await self.llm.embed(query)
        return self._retrieve(query, query_embedding, route_result)

    def _retrieve(
        self,
        query: str,
        query_embedding: list[float],
        route_result: RouteResult,
    ) -> list[dict]:
        """
        Retrieve relevant chunks from Milvus.

        Uses hybrid search (dense + BM25) by default.
        Applies metadata filters if the router extracted them.
        """
        # Build filters from route_result
        filters = {}
        if route_result.needs_filter and route_result.filters:
            for key in ("semester", "branch", "roll_no", "name", "course"):
                val = route_result.filters.get(key)
                if val is not None:
                    filters[key] = val

        # Run Milvus hybrid search (synchronous)
        return self.milvus.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            k=settings.RAG_TOP_K,
            filters=filters if filters else None,
        )

    def _assemble_context(
        self,
        chunks: list[dict],
        chat_history: list[dict] | None = None,
    ) -> str:
        """
        Assemble retrieved chunks into a context string for the LLM.

        Rules:
        - Number each chunk [1], [2], etc.
        - Include a metadata header line before each chunk text
        - Max 15 chunks in context
        - Truncate individual chunks to 500 chars if extremely long
        """
        max_chunks = 10
        max_chunk_chars = 500

        parts = [f"RETRIEVED STUDENT DATA ({min(len(chunks), max_chunks)} records):\n"]

        for i, chunk in enumerate(chunks[:max_chunks]):
            meta = chunk.get("metadata", {})
            header = (
                f"[{i + 1}] Student: {meta.get('name', 'N/A')} | "
                f"Roll: {meta.get('roll_no', 'N/A')} | "
                f"Branch: {meta.get('branch', 'N/A')} | "
                f"Sem: {meta.get('semester', 'N/A')} | "
                f"SGPA: {meta.get('sgpa', 'N/A')}"
            )

            text = chunk.get("text", "")
            if len(text) > max_chunk_chars:
                text = text[:max_chunk_chars] + "..."

            parts.append(f"\n{header}\n{text}\n")

        return "".join(parts)

    async def _generate_response(
        self,
        query: str,
        context_text: str,
        chat_history: list[dict] | None = None,
    ) -> str:
        """
        Generate a natural language response using the LLM with retrieved context.

        Uses chat() if chat_history is provided, otherwise generate().
        """
        system_prompt = RESPONSE_GENERATOR_PROMPT

        user_message = f"""Based on the following student data, answer the user's question.

{context_text}

Question: {query}"""

        if chat_history:
            messages = [{"role": "system", "content": system_prompt}]

            # Add recent chat history (last 6 messages max)
            for msg in (chat_history or [])[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": user_message})
            return await self.llm.chat(messages)
        else:
            return await self.llm.generate(prompt=user_message, system=system_prompt)
