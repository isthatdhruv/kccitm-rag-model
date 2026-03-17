"""Async Ollama REST API client for generation, chat, streaming, and embeddings.

This is the ONLY interface to the LLM in the project — all other modules import
and use this client.
"""

import asyncio
import json
import logging
from typing import AsyncGenerator

import httpx

from config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    """Async client for Ollama REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_HOST).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self.embed_model = embed_model or settings.OLLAMA_EMBED_MODEL
        self.timeout = httpx.Timeout(300.0, connect=10.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        format: str | None = None,
        model: str | None = None,
    ) -> str:
        """Generate a complete response (non-streaming)."""
        payload: dict = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                "num_predict": max_tokens or settings.LLM_MAX_TOKENS,
            },
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format

        data = await self._post("/api/generate", payload)
        return data.get("response", "")

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        format: str | None = None,
        model: str | None = None,
    ) -> str:
        """Chat completion with message history (non-streaming)."""
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                "num_predict": max_tokens or settings.LLM_MAX_TOKENS,
            },
        }
        if format:
            payload["format"] = format

        data = await self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    async def stream_chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion token by token."""
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature if temperature is not None else settings.LLM_TEMPERATURE,
                "num_predict": max_tokens or settings.LLM_MAX_TOKENS,
            },
        }
        url = f"{self.base_url}/api/chat"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("done"):
                        break
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token

    async def embed(self, text: str, model: str | None = None) -> list[float]:
        """Generate embedding vector for a single text (768-dim for nomic-embed-text)."""
        use_model = model or self.embed_model

        # Try newer /api/embed endpoint first
        try:
            data = await self._post(
                "/api/embed",
                {"model": use_model, "input": text},
            )
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
        except (httpx.HTTPStatusError, KeyError):
            pass

        # Fallback to older /api/embeddings endpoint
        data = await self._post(
            "/api/embeddings",
            {"model": use_model, "prompt": text},
        )
        return data.get("embedding", [])

    async def embed_batch(
        self, texts: list[str], model: str | None = None
    ) -> list[list[float]]:
        """Embed multiple texts with light parallelism (max 3 concurrent)."""
        sem = asyncio.Semaphore(3)

        async def _embed_one(t: str) -> list[float]:
            async with sem:
                return await self.embed(t, model=model)

        return await asyncio.gather(*[_embed_one(t) for t in texts])

    async def health_check(self) -> dict:
        """Check if Ollama is running and list available models."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                return {"status": "ok", "models": models}
        except httpx.ConnectError:
            return {"status": "error", "message": f"Cannot connect to Ollama at {self.base_url}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post(self, endpoint: str, payload: dict) -> dict:
        """POST with retry (3 attempts, exponential backoff)."""
        url = f"{self.base_url}{endpoint}"
        last_exc: Exception | None = None

        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 404 and "model" in payload:
                        model_name = payload["model"]
                        raise ValueError(
                            f"Model '{model_name}' not found in Ollama. Run: ollama pull {model_name}"
                        )
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(3**attempt)  # 1s, 3s
                    continue
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. Is it running?"
                ) from e
            except httpx.ReadTimeout as e:
                last_exc = e
                if attempt < 2:
                    await asyncio.sleep(3**attempt)
                    continue
                raise
            except (ValueError, httpx.HTTPStatusError):
                raise

        raise last_exc  # type: ignore[misc]
