"""
Advanced RAG optimization tests.
Tests HyDE, Multi-Query, Re-ranking, Compression individually + full pipeline.
Requires: MySQL + Milvus + Ollama all running with data.

Run: python -m tests.test_advanced_rag
"""
import asyncio
import time

from config import settings
from core.compressor import ContextualCompressor
from core.hyde import HyDEGenerator
from core.llm_client import OllamaClient
from core.multi_query import MultiQueryExpander
from core.reranker import ChunkReranker
from core.rag_pipeline import RAGPipeline
from core.router import QueryRouter
from db.milvus_client import MilvusSearchClient


async def test_hyde(llm: OllamaClient) -> None:
    """Test HyDE generation."""
    print("\n=== Test 1: HyDE Generation ===")
    hyde = HyDEGenerator(llm)

    query = "students struggling in programming"
    text, embedding = await hyde.generate_and_embed(query)

    print(f'  Query: "{query}"')
    print(f"  HyDE text ({len(text)} chars): {text[:150]}...")
    print(f"  Embedding dim: {len(embedding)}")

    assert len(text) > 30, "HyDE text too short"
    assert len(embedding) == settings.OLLAMA_EMBED_DIM, (
        f"Expected {settings.OLLAMA_EMBED_DIM}-dim, got {len(embedding)}"
    )

    text_lower = text.lower()
    has_academic_terms = any(
        term in text_lower
        for term in ["sgpa", "grade", "marks", "semester", "subject"]
    )
    print(f"  Contains academic terms: {has_academic_terms}")
    print("  \033[92m\u2713 HyDE generation works\033[0m")


async def test_multi_query(llm: OllamaClient) -> None:
    """Test multi-query expansion."""
    print("\n=== Test 2: Multi-Query Expansion ===")
    expander = MultiQueryExpander(llm)

    query = "top CSE students in semester 4"
    variants = await expander.expand(query)

    print(f'  Original: "{query}"')
    for i, v in enumerate(variants):
        print(f'  Variant {i + 1}: "{v}"')

    assert len(variants) >= 2, f"Expected 2-3 variants, got {len(variants)}"
    assert all(
        isinstance(v, str) and len(v) > 5 for v in variants
    ), "Variants must be non-empty strings"
    print(f"  \033[92m\u2713 Multi-query expansion works ({len(variants)} variants)\033[0m")


async def test_rrf() -> None:
    """Test Reciprocal Rank Fusion merge."""
    print("\n=== Test 3: Reciprocal Rank Fusion ===")

    list1 = [
        {"chunk_id": "A", "score": 0.9, "text": "chunk A"},
        {"chunk_id": "B", "score": 0.8, "text": "chunk B"},
        {"chunk_id": "C", "score": 0.7, "text": "chunk C"},
    ]
    list2 = [
        {"chunk_id": "B", "score": 0.95, "text": "chunk B"},
        {"chunk_id": "D", "score": 0.85, "text": "chunk D"},
        {"chunk_id": "A", "score": 0.75, "text": "chunk A"},
    ]

    merged = MultiQueryExpander.reciprocal_rank_fusion([list1, list2])

    ids = [m["chunk_id"] for m in merged]
    print("  List 1: A, B, C")
    print("  List 2: B, D, A")
    print(f"  Merged: {', '.join(ids)}")

    # B should rank first (appears in both lists at good positions)
    assert ids[0] == "B" or ids[1] == "B", "B should be in top 2 (appears in both lists)"
    # All 4 unique chunks should be present
    assert set(ids) == {"A", "B", "C", "D"}, "All chunks should be in merged results"
    # All should have rrf_score
    assert all("rrf_score" in m for m in merged), "All merged docs should have rrf_score"
    print("  \033[92m\u2713 RRF merge works correctly\033[0m")


async def test_reranker(llm: OllamaClient, milvus: MilvusSearchClient) -> None:
    """Test cross-encoder re-ranking."""
    print("\n=== Test 4: Cross-Encoder Re-ranking ===")

    query = "programming performance"
    embedding = await llm.embed(query)
    chunks = milvus.hybrid_search(query, embedding, k=10)

    if not chunks:
        print("  \033[91m\u2717 No chunks found \u2014 skipping reranker test\033[0m")
        return

    reranker = ChunkReranker()
    start = time.time()
    reranked = reranker.rerank(query, chunks, top_k=5)
    elapsed = (time.time() - start) * 1000

    print(f"  Input: {len(chunks)} chunks | Output: {len(reranked)} chunks | Time: {elapsed:.0f}ms")
    for i, chunk in enumerate(reranked[:3]):
        meta = chunk.get("metadata", {})
        print(
            f"  [{i + 1}] Score: {chunk.get('rerank_score', 0):.4f} | "
            f"{meta.get('name', 'N/A')} sem {meta.get('semester', '?')}"
        )

    assert len(reranked) == 5, f"Expected 5 results, got {len(reranked)}"
    assert all("rerank_score" in c for c in reranked), "All chunks should have rerank_score"
    scores = [c["rerank_score"] for c in reranked]
    assert scores == sorted(scores, reverse=True), "Scores should be descending"

    stats = reranker.get_score_stats(reranked)
    print(f"  Score stats: min={stats['min']:.4f} max={stats['max']:.4f} mean={stats['mean']:.4f}")
    print(f"  \033[92m\u2713 Re-ranking works ({elapsed:.0f}ms for {len(chunks)} pairs)\033[0m")


async def test_compressor(llm: OllamaClient, milvus: MilvusSearchClient) -> None:
    """Test contextual compression."""
    print("\n=== Test 5: Contextual Compression ===")

    query = "How did students perform in programming subjects?"
    embedding = await llm.embed(query)
    chunks = milvus.hybrid_search(query, embedding, k=5)

    if not chunks:
        print("  \033[91m\u2717 No chunks found \u2014 skipping compression test\033[0m")
        return

    compressor = ContextualCompressor(llm)
    compressed = await compressor.compress(query, chunks)

    savings = compressor.estimate_savings(chunks, compressed)
    print(f"  Original: {savings['original_tokens']} tokens in {len(chunks)} chunks")
    print(f"  Compressed: {savings['compressed_tokens']} tokens in {len(compressed)} chunks")
    print(f"  Saved: {savings['saved_tokens']} tokens ({savings['savings_percent']}%)")
    print(f"  Chunks removed (IRRELEVANT): {savings['chunks_removed']}")

    assert len(compressed) > 0, "At least some chunks should survive compression"
    assert savings["compressed_tokens"] <= savings["original_tokens"], (
        "Compressed should be <= original"
    )
    print(f"  \033[92m\u2713 Compression works ({savings['savings_percent']}% reduction)\033[0m")


async def test_full_pipeline_comparison(
    llm: OllamaClient, milvus: MilvusSearchClient
) -> None:
    """Compare basic vs optimized RAG pipeline on the same queries."""
    print("\n=== Test 6: Basic vs Optimized Pipeline Comparison ===")

    router = QueryRouter(llm)
    pipeline = RAGPipeline(llm, milvus)

    test_queries = [
        "students who scored poorly in programming subjects",
        "tell me about the CSE batch performance in semester 5",
        "KCS503 results",
    ]

    for query in test_queries:
        print(f'\n  Query: "{query}"')
        route_result = await router.route(query)

        # Basic RAG
        start = time.time()
        basic_result = await pipeline.run(
            query, route_result, use_optimizations=False
        )
        basic_time = (time.time() - start) * 1000

        # Optimized RAG
        start = time.time()
        opt_result = await pipeline.run(
            query, route_result, use_optimizations=True
        )
        opt_time = (time.time() - start) * 1000

        print(
            f"  Basic:     {basic_result.chunk_count} chunks | "
            f"{basic_time:.0f}ms | {len(basic_result.response)} chars"
        )
        print(
            f"  Optimized: {opt_result.chunk_count} chunks | "
            f"{opt_time:.0f}ms | {len(opt_result.response)} chars"
        )
        print(f"  Time delta: +{opt_time - basic_time:.0f}ms for optimizations")

    print("\n  \033[92m\u2713 Pipeline comparison complete\033[0m")


async def run_all() -> None:
    llm = OllamaClient()
    milvus = MilvusSearchClient(settings.MILVUS_HOST, settings.MILVUS_PORT)

    health = await llm.health_check()
    if health["status"] != "ok":
        print("\033[91m\u2717 Ollama not running\033[0m")
        return

    await test_hyde(llm)
    await test_multi_query(llm)
    await test_rrf()
    await test_reranker(llm, milvus)
    await test_compressor(llm, milvus)
    await test_full_pipeline_comparison(llm, milvus)

    print(f"\n{'=' * 60}")
    print("\033[92m\u2713 All advanced RAG tests complete!\033[0m")
    print(f"{'=' * 60}")

    print("\nTip: Run the interactive orchestrator to test the full system:")
    print("  python -m tests.test_orchestrator --interactive")


if __name__ == "__main__":
    asyncio.run(run_all())
