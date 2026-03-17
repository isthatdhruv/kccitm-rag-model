"""Milvus search wrapper — hybrid (dense + BM25), dense-only, keyword-only, FAQ."""

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker

from config import settings


class MilvusSearchClient:
    """Unified search interface over Milvus collections."""

    def __init__(
        self,
        host: str = settings.MILVUS_HOST,
        port: int = settings.MILVUS_PORT,
        collection: str = settings.MILVUS_COLLECTION,
    ):
        self.client = MilvusClient(uri=f"http://{host}:{port}")
        self.collection = collection

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        k: int = 30,
        filters: dict | None = None,
    ) -> list[dict]:
        """Dense (semantic) + sparse (BM25 keyword) search merged via RRF.

        Args:
            query_text: Raw text query (for BM25 sparse search).
            query_embedding: Dense embedding vector (768-dim).
            k: Number of results to return.
            filters: Optional metadata filters, e.g. {"semester": 4, "branch": "..."}.

        Returns:
            List of result dicts with chunk_id, text, score, and metadata.
        """
        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=k,
        )

        filter_expr = self._build_filter(filters)

        results = self.client.hybrid_search(
            collection_name=self.collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            filter=filter_expr,
            output_fields=self._output_fields(),
            limit=k,
        )
        return self._format_results(results)

    def dense_search(
        self,
        query_embedding: list[float],
        k: int = 30,
        filters: dict | None = None,
    ) -> list[dict]:
        """Dense-only vector search (for HyDE embeddings or when BM25 isn't needed)."""
        filter_expr = self._build_filter(filters)
        results = self.client.search(
            collection_name=self.collection,
            data=[query_embedding],
            anns_field="dense",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k,
            filter=filter_expr,
            output_fields=self._output_fields(),
        )
        return self._format_results(results)

    def keyword_search(
        self,
        query_text: str,
        k: int = 30,
        filters: dict | None = None,
    ) -> list[dict]:
        """BM25-only keyword search (for exact subject codes, roll numbers, etc.)."""
        filter_expr = self._build_filter(filters)
        results = self.client.search(
            collection_name=self.collection,
            data=[query_text],
            anns_field="sparse",
            search_params={"metric_type": "BM25"},
            limit=k,
            filter=filter_expr,
            output_fields=self._output_fields(),
        )
        return self._format_results(results)

    def search_faq(
        self,
        query_text: str,
        query_embedding: list[float],
        k: int = 1,
    ) -> dict | None:
        """Search the FAQ collection. Returns best match or None."""
        faq_collection = settings.MILVUS_FAQ_COLLECTION
        if not self.client.has_collection(faq_collection):
            return None

        stats = self.client.get_collection_stats(faq_collection)
        if stats.get("row_count", 0) == 0:
            return None

        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=k,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="sparse",
            param={"metric_type": "BM25"},
            limit=k,
        )

        results = self.client.hybrid_search(
            collection_name=faq_collection,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            output_fields=["question", "answer"],
            limit=k,
        )

        if results and results[0]:
            hit = results[0][0]
            return {
                "faq_id": hit.id,
                "question": hit.entity.get("question"),
                "answer": hit.entity.get("answer"),
                "score": hit.distance,
            }
        return None

    def get_collection_stats(self) -> dict:
        """Return collection stats: row_count, etc."""
        return self.client.get_collection_stats(self.collection)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _output_fields() -> list[str]:
        return [
            "text", "chunk_id", "roll_no", "name", "branch",
            "course", "semester", "sgpa", "session", "result_status", "gender",
        ]

    @staticmethod
    def _build_filter(filters: dict | None) -> str | None:
        """Convert a filter dict to a Milvus boolean expression string."""
        if not filters:
            return None
        conditions: list[str] = []
        if "semester" in filters:
            conditions.append(f'semester == {int(filters["semester"])}')
        if "branch" in filters:
            conditions.append(f'branch == "{filters["branch"]}"')
        if "roll_no" in filters:
            conditions.append(f'roll_no == "{filters["roll_no"]}"')
        if "name" in filters:
            conditions.append(f'name like "%{filters["name"]}%"')
        if "course" in filters:
            conditions.append(f'course == "{filters["course"]}"')
        return " and ".join(conditions) if conditions else None

    @staticmethod
    def _format_results(raw_results) -> list[dict]:
        """Normalise Milvus search results into clean dicts."""
        if not raw_results or not raw_results[0]:
            return []
        formatted: list[dict] = []
        for hit in raw_results[0]:
            formatted.append({
                "chunk_id": hit.id,
                "text": hit.entity.get("text"),
                "score": hit.distance,
                "metadata": {
                    "roll_no": hit.entity.get("roll_no"),
                    "name": hit.entity.get("name"),
                    "branch": hit.entity.get("branch"),
                    "course": hit.entity.get("course"),
                    "semester": hit.entity.get("semester"),
                    "sgpa": hit.entity.get("sgpa"),
                    "session": hit.entity.get("session"),
                    "result_status": hit.entity.get("result_status"),
                    "gender": hit.entity.get("gender"),
                },
            })
        return formatted
