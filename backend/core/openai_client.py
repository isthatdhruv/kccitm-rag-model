"""Lightweight OpenAI API client for fast structured tasks (SQL generation, routing).

Exposes the same generate() signature as OllamaClient so it can be used as a
drop-in replacement for latency-sensitive calls. Embeddings and streaming stay
on Ollama — this is only for fast JSON generation.
"""
from __future__ import annotations

import json
import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Async client for OpenAI chat completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model or settings.OPENAI_MODEL
        self.base_url = "https://api.openai.com/v1"
        self.timeout = httpx.Timeout(30.0, connect=10.0)

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Add it to .env or pass it directly."
            )

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        format: str | None = None,
        model: str | None = None,
    ) -> str:
        """Generate a response — matches OllamaClient.generate() signature."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return await self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            format=format,
            model=model,
        )

    async def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        max_tokens: int | None = None,
        format: str | None = None,
        model: str | None = None,
    ) -> str:
        """Chat completion — matches OllamaClient.chat() signature."""
        payload: dict = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else 0.3,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if format == "json":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def health_check(self) -> dict:
        """Verify the API key works."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                resp = await client.get(
                    f"{self.base_url}/models", headers=headers
                )
                resp.raise_for_status()
                return {"status": "ok", "provider": "openai", "model": self.model}
        except Exception as e:
            return {"status": "error", "message": str(e)}
