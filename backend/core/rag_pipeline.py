"""RAG pipeline: embed query -> Milvus hybrid search -> assemble context -> generate response.

Phase 4 basic pipeline + Phase 5 optimizations:
- HyDE (Hypothetical Document Embeddings)
- Multi-query expansion with RRF merge
- Cross-encoder re-ranking
- Contextual compression
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from config import settings
from core.compressor import ContextualCompressor
from core.hyde import HyDEGenerator
from core.llm_client import OllamaClient
from core.multi_query import MultiQueryExpander
from core.reranker import ChunkReranker
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
    RAG pipeline with optional advanced optimizations.

    When use_optimizations=True (default):
    HyDE -> multi-query expand -> retrieve (parallel) -> RRF merge -> re-rank -> compress -> generate

    When use_optimizations=False:
    Basic flow: embed query -> retrieve -> assemble -> generate
    """

    def __init__(self, llm: OllamaClient, milvus: MilvusSearchClient):
        self.llm = llm
        self.milvus = milvus
        # Optimization components
        self.hyde = HyDEGenerator(llm)
        self.multi_query = MultiQueryExpander(llm)
        self.reranker = ChunkReranker()
        self.compressor = ContextualCompressor(llm)

    async def run(
        self,
        query: str,
        route_result: RouteResult,
        chat_history: list[dict] | None = None,
        use_optimizations: bool = True,
    ) -> RAGResult:
        """
        Full RAG pipeline.

        Args:
            query: User's natural language question
            route_result: RouteResult from router (has filters, entities, intent)
            chat_history: Optional conversation history for context
            use_optimizations: If True, use HyDE + multi-query + re-rank + compress

        Returns:
            RAGResult with retrieved chunks, assembled context, and generated response
        """
        try:
            if use_optimizations:
                return await self._run_optimized(query, route_result, chat_history)
            else:
                return await self._run_basic(query, route_result, chat_history)
        except Exception as e:
            return RAGResult(success=False, error=f"RAG pipeline error: {str(e)}")

    async def _run_optimized(
        self,
        query: str,
        route_result: RouteResult,
        chat_history: list[dict] | None,
    ) -> RAGResult:
        """Full optimized pipeline with HyDE + multi-query + re-rank + compress."""
        filters = self._extract_filters(route_result)

        # Step 1: HyDE + Multi-query expansion (parallel — independent LLM calls)
        hyde_task = asyncio.create_task(self.hyde.generate_and_embed(query))
        expand_task = asyncio.create_task(self.multi_query.expand(query))

        (hyde_text, hyde_embedding), variants = await asyncio.gather(
            hyde_task, expand_task
        )

        # Step 2: Embed all variants
        all_queries = [query] + variants  # Original + up to 3 variants
        all_embeddings = [hyde_embedding]  # Use HyDE embedding for the original

        if variants:
            variant_embeddings = await self.llm.embed_batch(variants)
            all_embeddings.extend(variant_embeddings)

        # Step 3: Milvus hybrid search for each query variant
        all_results = []
        for q_text, q_embedding in zip(all_queries, all_embeddings):
            results = self.milvus.hybrid_search(
                query_text=q_text,
                query_embedding=q_embedding,
                k=20,
                filters=filters,
            )
            all_results.append(results)

        # Step 4: Merge via RRF
        merged = MultiQueryExpander.reciprocal_rank_fusion(all_results)
        candidates = merged[:30]  # Top 30 for re-ranking

        if not candidates:
            return RAGResult(
                success=True,
                chunks=[],
                chunk_count=0,
                response="I couldn't find relevant student data for your query. Could you rephrase?",
            )

        # Step 5: Cross-encoder re-ranking (30 -> top 10)
        reranked = self.reranker.rerank(query, candidates)

        # Step 6: Contextual compression
        compressed = await self.compressor.compress(query, reranked)

        # Step 7: Assemble context and generate
        context_text = self._assemble_context(compressed, chat_history)
        response = await self._generate_response(query, context_text, chat_history)

        return RAGResult(
            success=True,
            chunks=compressed,
            chunk_count=len(compressed),
            context_text=context_text,
            response=response,
        )

    async def _run_basic(
        self,
        query: str,
        route_result: RouteResult,
        chat_history: list[dict] | None,
    ) -> RAGResult:
        """Basic pipeline (Phase 4 logic) — used when optimizations are off."""
        query_embedding = await self.llm.embed(query)
        filters = self._extract_filters(route_result)

        chunks = self.milvus.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            k=settings.RAG_TOP_K,
            filters=filters,
        )

        if not chunks:
            return RAGResult(
                success=True,
                chunks=[],
                chunk_count=0,
                response="I couldn't find any relevant student data for your query. Could you rephrase or provide more details?",
            )

        context_text = self._assemble_context(chunks, chat_history)
        response = await self._generate_response(query, context_text, chat_history)

        return RAGResult(
            success=True,
            chunks=chunks,
            chunk_count=len(chunks),
            context_text=context_text,
            response=response,
        )

    async def retrieve_only(
        self,
        query: str,
        route_result: RouteResult,
        use_optimizations: bool = True,
    ) -> list[dict]:
        """
        Retrieve chunks without generating a response.
        Used by the HYBRID route where SQL pipeline also runs.
        """
        if not use_optimizations:
            query_embedding = await self.llm.embed(query)
            filters = self._extract_filters(route_result)
            return self.milvus.hybrid_search(
                query_text=query,
                query_embedding=query_embedding,
                k=settings.RAG_TOP_K,
                filters=filters,
            )

        # Optimized retrieval: HyDE + multi-query + re-rank + compress
        filters = self._extract_filters(route_result)

        hyde_task = asyncio.create_task(self.hyde.generate_and_embed(query))
        expand_task = asyncio.create_task(self.multi_query.expand(query))
        (hyde_text, hyde_embedding), variants = await asyncio.gather(
            hyde_task, expand_task
        )

        all_queries = [query] + variants
        all_embeddings = [hyde_embedding]
        if variants:
            all_embeddings.extend(await self.llm.embed_batch(variants))

        all_results = []
        for q_text, q_embedding in zip(all_queries, all_embeddings):
            results = self.milvus.hybrid_search(
                query_text=q_text,
                query_embedding=q_embedding,
                k=20,
                filters=filters,
            )
            all_results.append(results)

        merged = MultiQueryExpander.reciprocal_rank_fusion(all_results)
        reranked = self.reranker.rerank(query, merged[:30])
        compressed = await self.compressor.compress(query, reranked)
        return compressed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_filters(self, route_result: RouteResult) -> dict | None:
        """Extract Milvus filters from RouteResult."""
        if not route_result.needs_filter or not route_result.filters:
            return None
        filters = {}
        for key in ("semester", "branch", "roll_no", "name", "course"):
            val = route_result.filters.get(key)
            if val is not None:
                filters[key] = val
        return filters if filters else None

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
        - Max 10 chunks in context
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
        """Generate a natural language response using the LLM with retrieved context."""
        system_prompt = RESPONSE_GENERATOR_PROMPT

        user_message = f"""Based on the following student data, answer the user's question.

{context_text}

Question: {query}"""

        if chat_history:
            messages = [{"role": "system", "content": system_prompt}]

            for msg in (chat_history or [])[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

            messages.append({"role": "user", "content": user_message})
            return await self.llm.chat(messages)
        else:
            return await self.llm.generate(prompt=user_message, system=system_prompt)
