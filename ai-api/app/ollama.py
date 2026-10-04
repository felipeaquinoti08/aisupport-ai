"""Cliente do Ollama (rede interna): embeddings e geração de texto."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings
from .errors import LlmBusy, LlmUnavailable
from .logging_setup import log

logger = logging.getLogger("ollama")


@dataclass
class LlmResult:
    text: str
    model: str
    prompt_tokens: int
    output_tokens: int
    duration_ms: int


def _keep_alive(value: str) -> int | str:
    """O Ollama aceita número (segundos; -1 = sempre) ou duração com unidade ("10m")."""
    try:
        return int(value)
    except ValueError:
        return value


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class OllamaClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._s = settings
        self._client = httpx.AsyncClient(base_url=settings.ollama_url, transport=transport, timeout=settings.llm_timeout)
        # CPU sem GPU: uma geração por vez; demais aguardam numa fila limitada.
        self._llm_slot = asyncio.Semaphore(1)
        self._waiting = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            resp = await self._client.post(path, json=payload, timeout=timeout)
        except httpx.HTTPError as e:
            raise LlmUnavailable(f"{path}: {type(e).__name__}") from e
        if resp.status_code == 404:
            raise LlmUnavailable(f"modelo não encontrado: {payload.get('model')}")
        if resp.status_code >= 400:
            log(logger, logging.ERROR, "ollama_error", path=path, status=resp.status_code, body=resp.text[:300])
            raise LlmUnavailable(f"{path}: HTTP {resp.status_code}")
        return resp.json()

    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if not texts:
            return []
        instruction = self._s.embed_query_instruction
        if is_query and instruction:
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]
        data = await self._post(
            "/api/embed",
            {
                "model": self._s.embed_model,
                "input": texts,
                "truncate": True,
                "keep_alive": _keep_alive(self._s.ollama_keep_alive),
                "options": {"num_ctx": self._s.embed_num_ctx},
            },
            timeout=self._s.embed_timeout,
        )
        vectors = data.get("embeddings") or []
        if len(vectors) != len(texts):
            raise LlmUnavailable("resposta de embeddings incompleta")
        return [_normalize(v) for v in vectors]

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_output: bool | dict[str, Any] = False,
    ) -> LlmResult:
        if self._waiting >= self._s.llm_max_queue:
            raise LlmBusy("fila de geração cheia")
        self._waiting += 1
        try:
            await self._llm_slot.acquire()
        finally:
            self._waiting -= 1
        try:
            return await self._chat(messages, model, temperature, max_tokens, json_output)
        finally:
            self._llm_slot.release()

    async def _chat(self, messages, model, temperature, max_tokens, json_output) -> LlmResult:
        t0 = time.monotonic()
        model = model or self._s.llm_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": _keep_alive(self._s.ollama_keep_alive),
            "options": {
                "temperature": self._s.llm_temperature if temperature is None else temperature,
                # Mesma pergunta + mesmos documentos = mesma resposta
                "seed": self._s.llm_seed,
                "num_predict": max_tokens or self._s.llm_max_tokens,
                "num_ctx": self._s.llm_num_ctx,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        }
        if self._s.llm_num_thread > 0:
            payload["options"]["num_thread"] = self._s.llm_num_thread
        if json_output:
            # dict = JSON schema (geração restrita por gramática); True = JSON livre
            payload["format"] = json_output if isinstance(json_output, dict) else "json"
        data = await self._post("/api/chat", payload, timeout=self._s.llm_timeout)
        text = ((data.get("message") or {}).get("content") or "").strip()
        return LlmResult(
            text=text,
            model=data.get("model") or model,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            duration_ms=int((time.monotonic() - t0) * 1000),
        )

    async def list_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/tags", timeout=5)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise LlmUnavailable(f"/api/tags: {type(e).__name__}") from e
        return [m.get("name", "") for m in resp.json().get("models", [])]

    async def loaded_models(self) -> list[str]:
        try:
            resp = await self._client.get("/api/ps", timeout=5)
            resp.raise_for_status()
        except httpx.HTTPError:
            return []
        return [m.get("name", "") for m in resp.json().get("models", [])]

    @property
    def queue_size(self) -> int:
        return self._waiting
