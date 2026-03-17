"""Embedding generator: chunks → nomic-embed-text embeddings via Ollama.

Optimized for CPU machines:
- Uses Ollama's native batch /api/embed endpoint (one HTTP call per batch)
- Large batches (50-100 texts) to minimize HTTP overhead
- Checkpoint/resume: saves progress every batch so you don't lose work on crash

Usage:
    cd backend
    python -m ingestion.embedder
    python -m ingestion.embedder --resume   # Resume from checkpoint
"""

import asyncio
import json
import sys
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


async def embed_batch_native(
    texts: list[str],
    client: httpx.AsyncClient,
) -> list[list[float]]:
    """Embed a batch of texts using Ollama's native /api/embed endpoint.

    This sends all texts in a single HTTP call — much faster than
    one-at-a-time because Ollama batches the model forward pass.

    Retries up to 3 times with exponential backoff.
    """
    backoff = 2
    for attempt in range(3):
        try:
            response = await client.post(
                f"{settings.OLLAMA_HOST}/api/embed",
                json={"model": settings.OLLAMA_EMBED_MODEL, "input": texts},
                timeout=300.0,
            )
            response.raise_for_status()
            data = response.json()
            embeddings = data.get("embeddings", [])
            if len(embeddings) == len(texts):
                return embeddings
            # Fallback: if batch endpoint returns wrong count, try one-by-one
            raise ValueError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            if attempt == 2:
                raise RuntimeError(f"Batch embed failed after 3 attempts: {exc}") from exc
            print(f"{YELLOW}  Retry {attempt + 1}/3 (waiting {backoff}s): {exc}{RESET}")
            await asyncio.sleep(backoff)
            backoff *= 3
    return []


async def embed_all(
    chunks_path: Path | None = None,
    resume: bool = False,
) -> None:
    """Load chunks from JSONL, embed them in large batches, save results.

    Args:
        chunks_path: Path to chunks.jsonl (default: data/chunks.jsonl)
        resume: If True, skip chunks that already have embeddings in checkpoint
    """
    if chunks_path is None:
        chunks_path = settings.db_path("data/chunks.jsonl")

    emb_path = settings.db_path("data/embeddings.jsonl")
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    # Load chunks
    chunks: list[dict] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    total = len(chunks)
    print(f"  Loaded {total} chunks from {chunks_path}")

    # Load existing embeddings if resuming
    existing_ids: set[str] = set()
    if resume and emb_path.exists():
        with open(emb_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    existing_ids.add(obj["chunk_id"])
        print(f"  Resuming: {len(existing_ids)} chunks already embedded, {total - len(existing_ids)} remaining")

    # Filter out already-embedded chunks
    if existing_ids:
        remaining = [(i, c) for i, c in enumerate(chunks) if c["metadata"]["chunk_id"] not in existing_ids]
    else:
        remaining = list(enumerate(chunks))
        # Clear file if starting fresh
        emb_path.write_text("")

    if not remaining:
        print(f"{GREEN}✓ All chunks already embedded!{RESET}")
        _save_final_outputs(chunks, emb_path)
        return

    # Tunable via .env: EMBED_BATCH_SIZE (texts per call), EMBED_CONCURRENCY (parallel calls)
    batch_size = settings.EMBED_BATCH_SIZE
    concurrency = settings.EMBED_CONCURRENCY
    sem = asyncio.Semaphore(concurrency)
    start = time.time()
    done = len(existing_ids)

    # Open in append mode for checkpoint/resume
    mode = "a" if resume and existing_ids else "w"
    emb_file = open(emb_path, mode, encoding="utf-8")
    write_lock = asyncio.Lock()

    async def _process_batch(batch, client, pbar):
        async with sem:
            texts = [chunks[idx]["text"] for idx, _ in batch]
            batch_embeddings = await embed_batch_native(texts, client)

            async with write_lock:
                for (idx, chunk), emb in zip(batch, batch_embeddings):
                    cid = chunk["metadata"]["chunk_id"]
                    json.dump({"chunk_id": cid, "embedding": emb}, emb_file)
                    emb_file.write("\n")
                emb_file.flush()
            pbar.update(1)

    try:
        async with httpx.AsyncClient() as client:
            # Split into batches
            batches = [
                remaining[i : i + batch_size]
                for i in range(0, len(remaining), batch_size)
            ]
            total_batches = len(batches)

            with tqdm(total=total_batches, desc="Embedding", unit="batch") as pbar:
                if concurrency == 1:
                    # Sequential — simpler, no concurrency overhead
                    for batch in batches:
                        await _process_batch(batch, client, pbar)
                else:
                    # Parallel batches (GPU: 2-4 concurrent calls)
                    tasks = [_process_batch(b, client, pbar) for b in batches]
                    await asyncio.gather(*tasks)

    finally:
        emb_file.close()

    elapsed = time.time() - start
    minutes = elapsed / 60
    rate = len(remaining) / elapsed if elapsed > 0 else 0
    print(f"\n{GREEN}✓ Embedded {len(remaining)} chunks in {minutes:.1f} minutes ({rate:.1f} chunks/sec){RESET}")

    # Save final outputs (npy + chunk_ids.json)
    _save_final_outputs(chunks, emb_path)


def _save_final_outputs(chunks: list[dict], emb_path: Path) -> None:
    """Load all embeddings from JSONL and save as .npy + chunk_ids.json."""
    # Reload all embeddings (including previously checkpointed ones)
    embeddings_map: dict[str, list[float]] = {}
    with open(emb_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                embeddings_map[obj["chunk_id"]] = obj["embedding"]

    # Order by chunks list to maintain consistent ordering
    ordered_ids: list[str] = []
    ordered_embs: list[list[float]] = []
    for chunk in chunks:
        cid = chunk["metadata"]["chunk_id"]
        if cid in embeddings_map:
            ordered_ids.append(cid)
            ordered_embs.append(embeddings_map[cid])

    # Save embeddings.npy
    npy_path = emb_path.parent / "embeddings.npy"
    np.save(str(npy_path), np.array(ordered_embs, dtype=np.float32))

    # Save chunk_ids.json
    ids_path = emb_path.parent / "chunk_ids.json"
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(ordered_ids, f)

    print(f"  Saved {len(ordered_ids)} embeddings to {emb_path}")
    print(f"  Saved to {npy_path}")
    print(f"  Saved to {ids_path}")


if __name__ == "__main__":
    resume = "--resume" in sys.argv
    asyncio.run(embed_all(resume=resume))
