"""Cliente da API REST v2 do GLPI, usado SOMENTE para ler a Base de Conhecimento.

Autenticação OAuth2 (password grant) com uma conta de serviço somente-leitura.
O token é renovado automaticamente ao expirar ou ao receber 401.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings
from .errors import GlpiAuthError, GlpiNotFound, GlpiUnavailable
from .logging_setup import log

logger = logging.getLogger("glpi")

_CONTENT_RANGE = re.compile(r"(\d+)-(\d+)/(\d+)")


@dataclass
class KbArticle:
    id: int
    title: str
    content_html: str
    categories: list[dict[str, Any]] = field(default_factory=list)
    entity_id: int | None = None
    entity_name: str = ""
    is_recursive: bool = False
    is_faq: bool = False
    date_creation: str | None = None
    date_mod: str | None = None
    date_begin: str | None = None
    date_end: str | None = None

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "KbArticle":
        entity = data.get("entity") or {}
        cats = [
            {"id": c.get("id"), "name": c.get("name") or ""}
            for c in (data.get("categories") or [])
            if isinstance(c, dict)
        ]
        return cls(
            id=int(data["id"]),
            title=str(data.get("name") or "").strip(),
            content_html=str(data.get("content") or ""),
            categories=cats,
            entity_id=entity.get("id") if isinstance(entity, dict) else None,
            entity_name=(entity.get("name") or "") if isinstance(entity, dict) else "",
            is_recursive=bool(data.get("is_recursive")),
            is_faq=bool(data.get("is_faq")),
            date_creation=data.get("date_creation"),
            date_mod=data.get("date_mod"),
            date_begin=data.get("date_begin"),
            date_end=data.get("date_end"),
        )


class GlpiClient:
    API_PREFIX = "/api.php/v2"
    TOKEN_PATH = "/api.php/token"

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None):
        self._s = settings
        self._client = httpx.AsyncClient(
            base_url=settings.glpi_url,
            timeout=settings.glpi_timeout,
            verify=settings.glpi_verify,
            transport=transport,
            headers={"Accept": "application/json", "User-Agent": "aisupport-ai/1"},
            follow_redirects=False,
        )
        self._token: str | None = None
        self._token_exp = 0.0
        self._token_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def configured(self) -> bool:
        s = self._s
        return bool(s.glpi_oauth_client_id and s.glpi_username and s.glpi_password.get_secret_value())

    async def _get_token(self, force: bool = False) -> str:
        async with self._token_lock:
            if not force and self._token and time.monotonic() < self._token_exp - 30:
                return self._token
            if not self.configured:
                raise GlpiAuthError("credenciais da API do GLPI não configuradas")
            s = self._s
            try:
                resp = await self._client.post(
                    self.TOKEN_PATH,
                    data={
                        "grant_type": "password",
                        "client_id": s.glpi_oauth_client_id,
                        "client_secret": s.glpi_oauth_client_secret.get_secret_value(),
                        "username": s.glpi_username,
                        "password": s.glpi_password.get_secret_value(),
                        "scope": "api",
                    },
                )
            except httpx.HTTPError as e:
                raise GlpiUnavailable(f"falha de conexão ao obter token: {type(e).__name__}") from e
            if resp.status_code in (400, 401, 403):
                log(logger, logging.ERROR, "glpi_token_rejected", status=resp.status_code)
                raise GlpiAuthError(f"token recusado (HTTP {resp.status_code})")
            if resp.status_code >= 300:
                raise GlpiUnavailable(f"HTTP {resp.status_code} ao obter token")
            data = resp.json()
            token = data.get("access_token")
            if not token:
                raise GlpiAuthError("resposta de token sem access_token")
            self._token = token
            self._token_exp = time.monotonic() + float(data.get("expires_in") or 300)
            return token

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        for attempt in range(3):
            token = await self._get_token()
            try:
                resp = await self._client.get(
                    self.API_PREFIX + path,
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                )
            except httpx.TimeoutException as e:
                if attempt == 2:
                    raise GlpiUnavailable(f"timeout em {path}") from e
                await asyncio.sleep(1 + attempt)
                continue
            except httpx.HTTPError as e:
                raise GlpiUnavailable(f"falha de conexão em {path}: {type(e).__name__}") from e
            if resp.status_code == 401 and attempt < 2:
                self._token = None  # força novo token na próxima tentativa
                continue
            if resp.status_code in (502, 503, 504) and attempt < 2:
                await asyncio.sleep(1 + attempt)
                continue
            if resp.status_code in (401, 403):
                raise GlpiAuthError(f"acesso negado em {path} (HTTP {resp.status_code})")
            if resp.status_code == 404:
                raise GlpiNotFound(path)
            if resp.status_code >= 400:
                raise GlpiUnavailable(f"HTTP {resp.status_code} em {path}")
            return resp
        raise GlpiUnavailable(f"falha após novas tentativas em {path}")

    async def list_articles(self) -> list[KbArticle]:
        """Lista todos os artigos visíveis para a conta de serviço (paginado)."""
        articles: list[KbArticle] = []
        start = 0
        page = self._s.glpi_page_size
        while True:
            resp = await self._get("/Knowledgebase/Article", {"start": start, "limit": page, "sort": "id"})
            items = resp.json()
            if not isinstance(items, list):
                raise GlpiUnavailable("resposta inesperada ao listar artigos")
            for item in items:
                if isinstance(item, dict) and "id" in item:
                    articles.append(KbArticle.from_api(item))
            total = None
            m = _CONTENT_RANGE.search(resp.headers.get("Content-Range", ""))
            if m:
                total = int(m.group(3))
            start += page
            if not items or len(items) < page or (total is not None and start >= total):
                break
        return articles

    async def get_article(self, article_id: int) -> KbArticle | None:
        try:
            resp = await self._get(f"/Knowledgebase/Article/{int(article_id)}")
        except GlpiNotFound:
            return None
        return KbArticle.from_api(resp.json())

    async def check(self) -> dict[str, Any]:
        """Teste de conectividade: autentica e lê um artigo."""
        t0 = time.monotonic()
        await self._get("/Knowledgebase/Article", {"limit": 1})
        return {"latency_ms": int((time.monotonic() - t0) * 1000)}
