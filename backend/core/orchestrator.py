"""Master query orchestrator — routes queries to the appropriate pipeline and assembles responses.

Flow:
1. Router classifies query → SQL / RAG / HYBRID
2a. SQL route:    SQL pipeline generates + executes SQL → LLM summarizes results
2b. RAG route:    Milvus hybrid search → LLM generates from retrieved chunks
2c. HYBRID route: Both pipelines run in parallel, results merged → LLM generates
3. Return unified QueryResponse
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

from config import settings
from core.context_builder import ContextBuilder
from core.llm_client import OllamaClient
from core.rag_pipeline import RESPONSE_GENERATOR_PROMPT, RAGPipeline, RAGResult
from core.router import QueryRouter, RouteResult
from core.sql_pipeline import SQLPipeline, SQLResult
from db.milvus_client import MilvusSearchClient


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class QueryResponse:
    """Unified response from the orchestrator."""

    success: bool
    response: str = ""
    route_used: str = ""
    sql_result: SQLResult | None = None
    rag_result: RAGResult | None = None
    route_result: RouteResult | None = None
    total_time_ms: float = 0
    token_usage: dict = field(default_factory=dict)
    error: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """
    Master query orchestrator. Routes queries to the appropriate pipeline
    and assembles the final response.
    """

    def __init__(
        self,
        llm: OllamaClient,
        router: QueryRouter,
        sql_pipeline: SQLPipeline,
        rag_pipeline: RAGPipeline,
        milvus: MilvusSearchClient,
    ):
        self.llm = llm
        self.router = router
        self.sql_pipeline = sql_pipeline
        self.rag_pipeline = rag_pipeline
        self.milvus = milvus
        self.context_builder = ContextBuilder()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def process_query(
        self,
        query: str,
        chat_history: list[dict] | None = None,
    ) -> QueryResponse:
        """
        Process a user query end-to-end.

        Args:
            query: Natural language question
            chat_history: Optional conversation history

        Returns:
            QueryResponse with the answer, route used, and all metadata
        """
        start_time = time.time()

        try:
            # Step 1: Route the query
            route_result = await self.router.route(query, chat_history)

            # Step 2: Execute the appropriate pipeline
            if route_result.route == "SQL":
                response = await self._handle_sql(query, route_result, chat_history)
            elif route_result.route == "RAG":
                response = await self._handle_rag(query, route_result, chat_history)
            elif route_result.route == "HYBRID":
                response = await self._handle_hybrid(query, route_result, chat_history)
            else:
                # Unknown route — fall back to RAG
                route_result.route = "RAG"
                response = await self._handle_rag(query, route_result, chat_history)

            response.route_result = route_result
            response.route_used = route_result.route
            response.total_time_ms = (time.time() - start_time) * 1000

            return response

        except Exception as e:
            return QueryResponse(
                success=False,
                error=f"Orchestrator error: {str(e)}",
                total_time_ms=(time.time() - start_time) * 1000,
            )

    # ------------------------------------------------------------------
    # Route handlers
    # ------------------------------------------------------------------

    async def _handle_sql(
        self,
        query: str,
        route_result: RouteResult,
        chat_history: list[dict] | None,
    ) -> QueryResponse:
        """
        Handle SQL-routed queries.

        If SQL fails or returns 0 rows, automatically falls back to RAG.
        """
        sql_result = await self.sql_pipeline.run(query, route_result)

        if not sql_result.success:
            # SQL failed — fall back to RAG
            rag_result = await self.rag_pipeline.run(query, route_result, chat_history)
            return QueryResponse(
                success=rag_result.success,
                response=rag_result.response,
                route_used="RAG (SQL fallback)",
                sql_result=sql_result,
                rag_result=rag_result,
                metadata={"fallback": True, "sql_error": sql_result.error},
            )

        if sql_result.row_count == 0:
            # SQL returned no rows — try RAG for context
            rag_result = await self.rag_pipeline.run(query, route_result, chat_history)
            if rag_result.success and rag_result.chunk_count > 0:
                return QueryResponse(
                    success=True,
                    response=rag_result.response,
                    route_used="RAG (SQL empty)",
                    sql_result=sql_result,
                    rag_result=rag_result,
                    metadata={"fallback": True, "reason": "SQL returned 0 rows"},
                )

            return QueryResponse(
                success=True,
                response=(
                    "I couldn't find any matching data for your query. "
                    "The SQL query executed successfully but returned no results. "
                    "Could you try rephrasing or broadening your search?"
                ),
                route_used="SQL",
                sql_result=sql_result,
                metadata={"empty_result": True},
            )

        # SQL succeeded with results — generate natural language summary
        sql_context = self.context_builder.build_sql_context(sql_result)
        response_text = await self._generate_sql_summary(query, sql_context, chat_history)

        return QueryResponse(
            success=True,
            response=response_text,
            route_used="SQL",
            sql_result=sql_result,
        )

    async def _handle_rag(
        self,
        query: str,
        route_result: RouteResult,
        chat_history: list[dict] | None,
    ) -> QueryResponse:
        """Handle RAG-routed queries."""
        rag_result = await self.rag_pipeline.run(query, route_result, chat_history)

        return QueryResponse(
            success=rag_result.success,
            response=rag_result.response,
            route_used="RAG",
            rag_result=rag_result,
            error=rag_result.error if not rag_result.success else "",
        )

    async def _handle_hybrid(
        self,
        query: str,
        route_result: RouteResult,
        chat_history: list[dict] | None,
    ) -> QueryResponse:
        """
        Handle HYBRID-routed queries.

        Runs SQL and RAG retrieval in parallel, merges contexts,
        then generates a comprehensive response.
        """
        # Run both pipelines in parallel
        sql_task = asyncio.create_task(self.sql_pipeline.run(query, route_result))
        rag_task = asyncio.create_task(self.rag_pipeline.retrieve_only(query, route_result))

        sql_result, rag_chunks = await asyncio.gather(sql_task, rag_task)

        # Build combined context
        sql_context = ""
        if sql_result.success and sql_result.row_count > 0:
            sql_context = self.context_builder.build_sql_context(sql_result)

        rag_context = ""
        if rag_chunks:
            rag_context = self.context_builder.build_rag_context(rag_chunks)

        # Merge contexts
        if sql_context and rag_context:
            combined_context = f"{sql_context}\n\n{rag_context}"
        elif sql_context:
            combined_context = sql_context
        elif rag_context:
            combined_context = rag_context
        else:
            return QueryResponse(
                success=True,
                response=(
                    "I couldn't find sufficient data to answer this question "
                    "from either the database or the knowledge base."
                ),
                route_used="HYBRID",
                sql_result=sql_result,
            )

        # Generate response from combined context
        response_text = await self._generate_hybrid_response(
            query, combined_context, chat_history
        )

        rag_result = RAGResult(
            success=True,
            chunks=rag_chunks or [],
            chunk_count=len(rag_chunks or []),
            context_text=rag_context,
        )

        return QueryResponse(
            success=True,
            response=response_text,
            route_used="HYBRID",
            sql_result=sql_result,
            rag_result=rag_result,
        )

    # ------------------------------------------------------------------
    # LLM generation helpers
    # ------------------------------------------------------------------

    async def _generate_sql_summary(
        self,
        query: str,
        sql_context: str,
        chat_history: list[dict] | None,
    ) -> str:
        """Generate natural language summary of SQL results."""
        system_prompt = RESPONSE_GENERATOR_PROMPT.replace(
            "CONTEXT TYPE: RAG (Retrieved student data chunks)",
            "CONTEXT TYPE: SQL (Database query results)",
        )
        prompt = (
            "Based on the following database query results, "
            f"answer the user's question in natural language.\n\n"
            f"{sql_context}\n\nQuestion: {query}"
        )

        if chat_history:
            messages = [{"role": "system", "content": system_prompt}]
            for msg in (chat_history or [])[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})
            return await self.llm.chat(messages)

        return await self.llm.generate(prompt=prompt, system=system_prompt)

    async def _generate_hybrid_response(
        self,
        query: str,
        combined_context: str,
        chat_history: list[dict] | None,
    ) -> str:
        """Generate response from combined SQL + RAG context."""
        system_prompt = RESPONSE_GENERATOR_PROMPT.replace(
            "CONTEXT TYPE: RAG (Retrieved student data chunks)",
            "CONTEXT TYPE: HYBRID (Database query results + Retrieved student data)",
        )
        prompt = (
            "Based on the following data (from both database queries and student records), "
            f"provide a comprehensive answer.\n\n"
            f"{combined_context}\n\nQuestion: {query}"
        )

        if chat_history:
            messages = [{"role": "system", "content": system_prompt}]
            for msg in (chat_history or [])[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})
            return await self.llm.chat(messages)

        return await self.llm.generate(prompt=prompt, system=system_prompt)

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def process_query_stream(
        self,
        query: str,
        chat_history: list[dict] | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process query with streaming response.

        Routing + retrieval happen first (non-streaming),
        then only the final LLM generation is streamed.
        """
        # Step 1: Route (non-streaming, fast)
        route_result = await self.router.route(query, chat_history)
        filters = route_result.filters if route_result.needs_filter else None

        # Step 2: Retrieve/execute (non-streaming)
        context_type = route_result.route

        if route_result.route == "SQL":
            sql_result = await self.sql_pipeline.run(query, route_result)
            if sql_result.success and sql_result.row_count > 0:
                context = self.context_builder.build_sql_context(sql_result)
            else:
                # Fallback to RAG
                query_embedding = await self.llm.embed(query)
                chunks = self.milvus.hybrid_search(
                    query, query_embedding, k=settings.RAG_TOP_K, filters=filters
                )
                context = self.context_builder.build_rag_context(chunks)
                context_type = "RAG"

        elif route_result.route == "RAG":
            query_embedding = await self.llm.embed(query)
            chunks = self.milvus.hybrid_search(
                query, query_embedding, k=settings.RAG_TOP_K, filters=filters
            )
            context = self.context_builder.build_rag_context(chunks)

        else:
            # HYBRID — get both, merge context
            sql_task = asyncio.create_task(self.sql_pipeline.run(query, route_result))
            embed_task = asyncio.create_task(self.llm.embed(query))
            sql_result, query_embedding = await asyncio.gather(sql_task, embed_task)

            chunks = self.milvus.hybrid_search(
                query, query_embedding, k=settings.RAG_TOP_K, filters=filters
            )
            sql_ctx = (
                self.context_builder.build_sql_context(sql_result)
                if sql_result.success
                else ""
            )
            rag_ctx = self.context_builder.build_rag_context(chunks) if chunks else ""
            context = f"{sql_ctx}\n\n{rag_ctx}".strip()

        # Step 3: Stream the LLM response
        system_prompt = RESPONSE_GENERATOR_PROMPT.replace(
            "CONTEXT TYPE: RAG (Retrieved student data chunks)",
            f"CONTEXT TYPE: {context_type}",
        )
        prompt = (
            f"Based on the following data, answer the user's question.\n\n"
            f"{context}\n\nQuestion: {query}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for msg in (chat_history or [])[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": prompt})

        async for token in self.llm.stream_chat(messages):
            yield token
