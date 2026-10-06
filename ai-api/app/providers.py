"""Geração em provedor externo (OpenAI, Azure OpenAI/AI Foundry, API compatível
com OpenAI, Anthropic). A busca continua local: embeddings no Ollama e
artigos no Qdrant, já filtrados pelas permissões do GLPI.

A configuração chega do plugin a cada requisição (a chave fica criptografada
no GLPI) e nunca é gravada nem registrada em log aqui.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import anthropic
import httpx

from .logging_setup import log
from .ollama import LlmResult

logger = logging.getLogger("provider")

KINDS = ("openai", "azure", "compatible", "anthropic")
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}
# Recusas de segurança dos modelos atuais da Anthropic voltam com fallback no próprio servidor
_ANTHROPIC_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


class ProviderError(Exception):
    """Falha do provedor externo. `code` é estável e vai para o log e o painel."""

    def __init__(self, code: str, status: int = 0, detail: str = ""):
        super().__init__(code)
        self.code = code
        self.status = status
        self.detail = detail[:300]


@dataclass
class ProviderConfig:
    kind: str
    api_key: str
    model: str
    base_url: str = ""
    api_version: str = ""
    effort: str = ""
    timeout: float = 45.0
    mask_pii: bool = True
    fallback_local: bool = True

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.model}"


def strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """JSON schema estrito (todas as chaves obrigatórias, sem extras)."""
    out = dict(schema)
    out["additionalProperties"] = False
    out["required"] = list(schema.get("properties", {}).keys())
    return out


class ExternalLLM:
    def __init__(self, http: httpx.AsyncClient | None = None, anthropic_factory=None):
        self._http = http or httpx.AsyncClient(follow_redirects=False)
        # Injetável nos testes; por padrão o SDK oficial
        self._anthropic_factory = anthropic_factory or (
            lambda cfg: anthropic.AsyncAnthropic(
                api_key=cfg.api_key,
                base_url=cfg.base_url or None,
                timeout=cfg.timeout,
                max_retries=1,
            )
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(
        self,
        cfg: ProviderConfig,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float | None = None,
    ) -> LlmResult:
        t0 = time.monotonic()
        if cfg.kind == "anthropic":
            text, usage = await self._anthropic(cfg, messages, schema, max_tokens)
        elif cfg.kind in ("openai", "azure", "compatible"):
            text, usage = await self._openai_like(cfg, messages, schema, max_tokens, temperature)
        else:
            raise ProviderError("invalid_provider")
        ms = int((time.monotonic() - t0) * 1000)
        log(logger, logging.INFO, "provider_call", provider=cfg.kind, model=cfg.model, ms=ms,
            prompt_tokens=usage[0], output_tokens=usage[1])
        return LlmResult(text=text, model=cfg.label, prompt_tokens=usage[0], output_tokens=usage[1], duration_ms=ms)

    # --- OpenAI / Azure / compatível ----------------------------------------------------

    def _openai_target(self, cfg: ProviderConfig) -> tuple[str, dict[str, str], bool]:
        """URL, cabeçalhos e se o modelo vai no corpo."""
        if cfg.kind == "azure":
            base = cfg.base_url.rstrip("/")
            headers = {"api-key": cfg.api_key}
            if base.endswith("/models"):
                # Azure AI Foundry (Model Inference API)
                return f"{base}/chat/completions?api-version={cfg.api_version or '2024-05-01-preview'}", headers, True
            if cfg.api_version:
                # Azure OpenAI clássico: o "modelo" é o nome do deployment
                return f"{base}/openai/deployments/{cfg.model}/chat/completions?api-version={cfg.api_version}", headers, False
            if not base.endswith("/openai/v1"):
                base = f"{base}/openai/v1"
            return f"{base}/chat/completions", headers, True
        base = (cfg.base_url or DEFAULT_BASE_URLS.get(cfg.kind, "")).rstrip("/")
        return f"{base}/chat/completions", {"Authorization": f"Bearer {cfg.api_key}"}, True

    async def _openai_like(self, cfg, messages, schema, max_tokens, temperature) -> tuple[str, tuple[int, int]]:
        url, headers, model_in_body = self._openai_target(cfg)
        body: dict[str, Any] = {"messages": messages, "max_completion_tokens": max_tokens}
        if model_in_body:
            body["model"] = cfg.model
        if temperature is not None:
            body["temperature"] = temperature
        formats: list[dict[str, Any] | None] = [None]
        if schema:
            formats = [
                {"type": "json_schema", "json_schema": {"name": "resposta", "strict": True, "schema": strict_schema(schema)}},
                {"type": "json_object"},
                None,
            ]
        last: ProviderError | None = None
        for fmt in formats:
            payload = dict(body)
            if fmt:
                payload["response_format"] = fmt
            for _ in range(3):
                try:
                    data = await self._post(url, headers, payload, cfg.timeout)
                    choice = (data.get("choices") or [{}])[0]
                    text = ((choice.get("message") or {}).get("content") or "").strip()
                    if choice.get("finish_reason") == "content_filter" or not text:
                        raise ProviderError("refused" if choice.get("finish_reason") == "content_filter" else "empty_response")
                    usage = data.get("usage") or {}
                    return text, (int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0))
                except ProviderError as e:
                    last = e
                    if e.code != "bad_request":
                        raise
                    detail = e.detail.lower()
                    # Modelos que não aceitam temperatura ou max_completion_tokens
                    if "temperature" in detail and "temperature" in payload:
                        payload.pop("temperature")
                        continue
                    if "max_completion_tokens" in detail and "max_completion_tokens" in payload:
                        payload["max_tokens"] = payload.pop("max_completion_tokens")
                        continue
                    break
            # Formato de saída não suportado: tenta o próximo, mais simples
        raise last or ProviderError("bad_request")

    async def _post(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        try:
            resp = await self._http.post(url, headers={**headers, "Content-Type": "application/json"}, json=payload, timeout=timeout)
        except httpx.TimeoutException as e:
            raise ProviderError("timeout") from e
        except httpx.HTTPError as e:
            raise ProviderError("unreachable", detail=type(e).__name__) from e
        if resp.status_code >= 400:
            raise ProviderError(_status_code(resp.status_code), resp.status_code, _error_detail(resp))
        try:
            return resp.json()
        except ValueError as e:
            raise ProviderError("invalid_response", resp.status_code) from e

    # --- Anthropic (SDK oficial) -------------------------------------------------------

    async def _anthropic(self, cfg, messages, schema, max_tokens) -> tuple[str, tuple[int, int]]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        turns = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        output_config: dict[str, Any] = {}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": strict_schema(schema)}
        if cfg.effort:
            output_config["effort"] = cfg.effort
        params: dict[str, Any] = {"model": cfg.model, "max_tokens": max_tokens, "messages": turns}
        if system:
            params["system"] = system
        if output_config:
            params["output_config"] = output_config
        client = self._anthropic_factory(cfg)
        try:
            if cfg.model in _ANTHROPIC_FALLBACK_MODELS and not cfg.base_url:
                response = await client.beta.messages.create(
                    **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                )
            else:
                response = await client.messages.create(**params)
        except anthropic.AuthenticationError as e:
            raise ProviderError("auth_failed", 401) from e
        except anthropic.PermissionDeniedError as e:
            raise ProviderError("forbidden", 403) from e
        except anthropic.NotFoundError as e:
            raise ProviderError("model_not_found", 404) from e
        except anthropic.RateLimitError as e:
            raise ProviderError("rate_limited", 429) from e
        except anthropic.BadRequestError as e:
            raise ProviderError("bad_request", 400, str(e)) from e
        except anthropic.APITimeoutError as e:
            raise ProviderError("timeout") from e
        except anthropic.APIStatusError as e:
            raise ProviderError(_status_code(e.status_code), e.status_code) from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("unreachable") from e
        finally:
            await client.close()
        if response.stop_reason == "refusal":
            raise ProviderError("refused")
        text = next((b.text for b in response.content if getattr(b, "type", "") == "text"), "").strip()
        if not text:
            raise ProviderError("empty_response")
        usage = response.usage
        return text, (int(getattr(usage, "input_tokens", 0) or 0), int(getattr(usage, "output_tokens", 0) or 0))


def _status_code(status: int) -> str:
    return {400: "bad_request", 401: "auth_failed", 403: "forbidden", 404: "model_not_found", 429: "rate_limited"}.get(
        status, "provider_error" if status >= 500 else f"http_{status}"
    )


def _error_detail(resp: httpx.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error") if isinstance(data, dict) else None
        if isinstance(err, dict):
            return str(err.get("message") or err.get("code") or "")
        return str(err or data.get("message") or "")
    except ValueError:
        return resp.text[:300]


# --- Dados pessoais --------------------------------------------------------------------

_PII_PATTERNS = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("CNPJ", re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b")),
    ("CPF", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b")),
    ("IP", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")),
    ("TELEFONE", re.compile(r"(?<!\w)(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}(?!\w)")),
]


class PiiMasker:
    """Troca dados pessoais por marcadores ([EMAIL_1]...) antes de enviar ao
    provedor externo e restaura os valores na resposta. O mesmo valor recebe
    sempre o mesmo marcador dentro de uma requisição."""

    def __init__(self) -> None:
        self._by_value: dict[str, str] = {}
        self._by_tag: dict[str, str] = {}
        self._count: dict[str, int] = {}

    def mask(self, text: str) -> str:
        for kind, pattern in _PII_PATTERNS:
            text = pattern.sub(lambda m, k=kind: self._tag(k, m.group(0)), text)
        return text

    def _tag(self, kind: str, value: str) -> str:
        if value in self._by_value:
            return self._by_value[value]
        self._count[kind] = self._count.get(kind, 0) + 1
        tag = f"[{kind}_{self._count[kind]}]"
        self._by_value[value] = tag
        self._by_tag[tag] = value
        return tag

    def unmask(self, text: str) -> str:
        if not self._by_tag:
            return text
        return re.sub(r"\[[A-Z]+_\d+\]", lambda m: self._by_tag.get(m.group(0), m.group(0)), text)

    @property
    def count(self) -> int:
        return len(self._by_tag)


def mask_messages(messages: list[dict[str, str]], masker: PiiMasker) -> list[dict[str, str]]:
    # O prompt do sistema é nosso (sem dados pessoais); só o conteúdo do usuário é mascarado
    return [m if m["role"] == "system" else {**m, "content": masker.mask(m["content"])} for m in messages]


def unmask_json_text(text: str, masker: PiiMasker) -> str:
    """Restaura dentro das strings do JSON gerado, mantendo o JSON válido."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return masker.unmask(text)

    def walk(v):
        if isinstance(v, str):
            return masker.unmask(v)
        if isinstance(v, dict):
            return {k: walk(x) for k, x in v.items()}
        if isinstance(v, list):
            return [walk(x) for x in v]
        return v

    return json.dumps(walk(data), ensure_ascii=False)
