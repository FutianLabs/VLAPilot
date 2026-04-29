"""LLM Client - unified interface for all providers."""

from __future__ import annotations

import httpx
import os
from typing import Any


class LLMClient:
    """Unified LLM client using OpenAI-compatible API."""

    def __init__(self, provider: str, model: str, api_base: str, api_key: str = "", proxy: str | None | bool = None):
        self.provider = provider
        self.model = model
        self.api_base = api_base
        self.api_key = api_key

        if proxy is False:
            actual_proxy = None
        elif proxy is None:
            actual_proxy = os.getenv("https_proxy") or os.getenv("http_proxy")
        else:
            actual_proxy = proxy

        self.client = httpx.AsyncClient(timeout=90.0, proxy=actual_proxy)

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> dict:
        """Call LLM chat completion API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **kwargs
        }

        if tools:
            payload["tools"] = tools

        url = f"{self.api_base}/chat/completions"

        # Debug log
        from loguru import logger
        logger.debug(f"LLM request: {self.provider} {self.model}")
        logger.debug(f"URL: {url}")
        logger.debug(f"Payload keys: {list(payload.keys())}")

        response = await self.client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
