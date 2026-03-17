"""Embedding generator: chunks → nomic-embed-text embeddings via Ollama.

Usage:
    cd backend
    python -m ingestion.embedder
"""

import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np
from tqdm import tqdm

from config import settings

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


async def embed_text(text: str, client: httpx.AsyncClient) -> list[float]:
    """Embed a single text using Ollama's /api/embeddings endpoint.

    Retries up to 3 times with exponential backoff (1s, 3s, 9s).
    """
    backoff = 1
    for attempt in range(3):
        try:
            response = await client.post(
                f"{settings.OLLAMA_HOST}/api/embeddings",
                json={"model": settings.OLLAMA_EMBED_MODEL, "prompt": text},
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["embedding"]
        except (httpx.HTTPError, KeyError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Failed to embed text after 3 attempts: {exc}") from exc
            print(f"{YELLOW}  Retry {attempt + 1}/3 (waiting {backoff}s): {exc}{RESET}")
            await asyncio.sleep(backoff)
            backoff *= 3
    return []  # unreachable, but keeps type checkers happy


async def embed_all(chunks_path: Path | None = None) -> None:
    """Load chunks from JSONL, embed them all, and save results."""
    if chunks_path is None:
        chunks_path = settings.db_path("data/chunks.jsonl")

    # Load chunks
    chunks: list[dict] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    total = len(chunks)
    print(f"  Loaded {total} chunks from {chunks_path}")

    embeddings: list[list[float]] = []
    chunk_ids: list[str] = []
    start = time.time()

    async with httpx.AsyncClient() as client:
        # Process in batches of 10 for progress display
        batch_size = 10
        for i in tqdm(range(0, total, batch_size), desc="Embedding", unit="batch"):
            batch = chunks[i : i + batch_size]
            tasks = [embed_text(c["text"], client) for c in batch]
            batch_embeddings = await asyncio.gather(*tasks)
            for chunk, emb in zip(batch, batch_embeddings):
                embeddings.append(emb)
                chunk_ids.append(chunk["metadata"]["chunk_id"])

    elapsed = time.time() - start

    # Save embeddings.jsonl
    emb_path = settings.db_path("data/embeddings.jsonl")
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    with open(emb_path, "w", encoding="utf-8") as f:
        for cid, emb in zip(chunk_ids, embeddings):
            json.dump({"chunk_id": cid, "embedding": emb}, f)
            f.write("\n")

    # Save embeddings.npy
    npy_path = settings.db_path("data/embeddings.npy")
    np.save(str(npy_path), np.array(embeddings, dtype=np.float32))

    # Save chunk_ids.json
    ids_path = settings.db_path("data/chunk_ids.json")
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f)

    minutes = elapsed / 60
    print(f"\n{GREEN}✓ Embedded {total} chunks in {minutes:.1f} minutes.{RESET}")
    print(f"  Saved to {emb_path}")
    print(f"  Saved to {npy_path}")
    print(f"  Saved to {ids_path}")


if __name__ == "__main__":
    asyncio.run(embed_all())
