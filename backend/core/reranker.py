"""Cross-encoder re-ranking for precision retrieval.

Unlike bi-encoders (which encode query and document separately),
cross-encoders process the query-document pair TOGETHER, giving
much more accurate relevance scores at the cost of being slower.

Pipeline position: after retrieval (30 candidates) -> re-rank -> top 10.
Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB, runs on CPU)
Latency: ~50-80ms for 30 pairs
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from config import settings


class ChunkReranker:
    """Re-rank retrieved chunks using a cross-encoder model (singleton)."""

    _model: CrossEncoder | None = None
    _model_name: str | None = None

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.RERANKER_MODEL

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model (loads once, cached as class var)."""
        if ChunkReranker._model is None or ChunkReranker._model_name != self.model_name:
            print(f"Loading re-ranker model: {self.model_name}...")
            ChunkReranker._model = CrossEncoder(self.model_name)
            ChunkReranker._model_name = self.model_name
            print("Re-ranker loaded.")
        return ChunkReranker._model

    def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_k: int | None = None,
        text_field: str = "text",
    ) -> list[dict]:
        """
        Re-rank chunks by cross-encoder relevance score.

        Args:
            query: User's query
            chunks: List of chunk dicts (must have a text field)
            top_k: Number of top results to return (default: settings.RAG_RERANK_TOP_K)
            text_field: Key in chunk dict that contains the text

        Returns:
            Top-k chunks sorted by cross-encoder score (highest first).
        """
        top_k = top_k or settings.RAG_RERANK_TOP_K

        if not chunks:
            return []

        # Build query-document pairs
        texts = [chunk.get(text_field, "") for chunk in chunks]
        pairs = [[query, text] for text in texts]

        # Score all pairs in batch
        scores = self.model.predict(pairs)

        # Attach scores and sort
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            chunk_copy = chunk.copy()
            chunk_copy["rerank_score"] = float(score)
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_chunks[:top_k]

    def get_score_stats(self, chunks: list[dict]) -> dict:
        """Get statistics about re-ranking scores for monitoring."""
        if not chunks:
            return {"min": 0, "max": 0, "mean": 0, "count": 0}

        scores = [c.get("rerank_score", 0) for c in chunks]
        return {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "count": len(scores),
        }
