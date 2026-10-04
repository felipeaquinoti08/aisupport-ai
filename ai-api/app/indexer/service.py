"""Indexação da Base de Conhecimento do GLPI no Qdrant.

- full(): lê todos os artigos via API v2, indexa numa coleção nova e troca o alias.
- sync(): compara o hash de cada artigo com o indexado; reindexa alterados/novos e
  remove os que deixaram de existir (ou deixaram de ser visíveis para a conta de serviço).

O GLPI é sempre a origem: nada aqui é editável manualmente.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from .. import sparse
from ..config import Settings
from ..errors import SyncAlreadyRunning
from ..glpi_client import GlpiClient, KbArticle
from ..logging_setup import log
from ..ollama import OllamaClient
from ..text import chunk_text, html_to_text, looks_like_injection
from ..vectorstore import VectorStore, validity_window

logger = logging.getLogger("indexer")

# Mude quando a forma de gerar trechos/payload mudar: força reindexação no sync.
PIPELINE_VERSION = "1"
_EMBED_BATCH = 8
_LOCK_PATH = os.environ.get("INDEX_LOCK_PATH", "/tmp/aisupport-index.lock")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class IndexerService:
    def __init__(self, settings: Settings, glpi: GlpiClient, ollama: OllamaClient, store: VectorStore):
        self._s = settings
        self._glpi = glpi
        self._ollama = ollama
        self._store = store
        self._lock = asyncio.Lock()
        self.running: str | None = None

    @asynccontextmanager
    async def _exclusive(self, kind: str):
        """Exclusão mútua dentro do processo e entre processos (CLI x API)."""
        if self._lock.locked():
            raise SyncAlreadyRunning(self.running or "indexação")
        async with self._lock:
            fh = open(_LOCK_PATH, "w")
            try:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as e:
                    raise SyncAlreadyRunning("indexação em outro processo") from e
                self.running = kind
                yield
            finally:
                self.running = None
                fh.close()

    # --- Montagem dos trechos ---------------------------------------------------------

    def article_hash(self, a: KbArticle) -> str:
        s = self._s
        raw = json.dumps(
            [PIPELINE_VERSION, s.embed_model, s.rag_chunk_size, s.rag_chunk_overlap, a.title, a.content_html,
             a.categories, a.entity_id, a.is_recursive, a.is_faq, a.date_begin, a.date_end],
            ensure_ascii=False, sort_keys=True, default=str,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _chunks(self, a: KbArticle, content_hash: str) -> list[dict[str, Any]]:
        text = html_to_text(a.content_html)
        pieces = chunk_text(text, self._s.rag_chunk_size, self._s.rag_chunk_overlap)
        if not pieces:
            return []
        suspicious = looks_like_injection(a.title) or any(looks_like_injection(p) for p in pieces)
        valid_from, valid_until = validity_window(a.date_begin, a.date_end, self._s.glpi_timezone)
        categories = [c["name"] for c in a.categories if c.get("name")]
        base = {
            "article_id": a.id,
            "title": a.title,
            "categories": categories,
            "category_ids": [c["id"] for c in a.categories if c.get("id") is not None],
            "entity_id": a.entity_id,
            "entity_name": a.entity_name,
            "is_recursive": a.is_recursive,
            "is_faq": a.is_faq,
            "date_creation": a.date_creation,
            "date_mod": a.date_mod,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "url": f"{self._s.glpi_url}/front/knowbaseitem.form.php?id={a.id}",
            "content_hash": content_hash,
            "suspicious": suspicious,
            "indexed_at": _now_iso(),
            "chunks_total": len(pieces),
        }
        header = a.title + (f"\nCategoria: {', '.join(categories)}" if categories else "")
        return [
            {"payload": {**base, "chunk_index": i, "text": piece}, "embed_text": f"{header}\n{piece}"}
            for i, piece in enumerate(pieces)
        ]

    async def _index_article(self, collection: str, a: KbArticle, content_hash: str) -> dict[str, Any]:
        chunks = self._chunks(a, content_hash)
        if not chunks:
            await self._store.delete_article(collection, a.id)
            return {"chunks": 0, "suspicious": False}
        vectors: list[list[float]] = []
        for i in range(0, len(chunks), _EMBED_BATCH):
            batch = chunks[i:i + _EMBED_BATCH]
            vectors.extend(await self._ollama.embed([c["embed_text"] for c in batch]))
        for c, vec in zip(chunks, vectors):
            c["dense"] = vec
            c["sparse"] = sparse.document_vector(c["embed_text"])
        await self._store.upsert_article(collection, a.id, chunks)
        suspicious = chunks[0]["payload"]["suspicious"]
        if suspicious:
            log(logger, logging.WARNING, "article_flagged_suspicious", article_id=a.id)
        return {"chunks": len(chunks), "suspicious": suspicious}

    async def _embedding_dim(self) -> int:
        return len((await self._ollama.embed(["dimensão"]))[0])

    # --- Operações ---------------------------------------------------------------------

    async def full(self) -> dict[str, Any]:
        async with self._exclusive("full"):
            return await self._full()

    async def _full(self) -> dict[str, Any]:
        t0 = time.monotonic()
        articles = await self._glpi.list_articles()
        dim = await self._embedding_dim()
        collection = await self._store.create_collection(dim)
        indexed = chunks = 0
        suspicious: list[dict[str, Any]] = []
        try:
            for a in articles:
                r = await self._index_article(collection, a, self.article_hash(a))
                if r["chunks"]:
                    indexed += 1
                    chunks += r["chunks"]
                if r["suspicious"]:
                    suspicious.append({"id": a.id, "title": a.title})
        except Exception:
            await self._store.drop_collection(collection)
            raise
        await self._store.activate(collection)
        result = {
            "mode": "full", "articles_seen": len(articles), "articles_indexed": indexed,
            "chunks": chunks, "deleted": 0, "suspicious": len(suspicious),
            "duration_ms": int((time.monotonic() - t0) * 1000),
        }
        now = _now_iso()
        await self._store.set_state(
            last_full_at=now, last_sync_at=now, last_result=result, last_error=None,
            embed_model=self._s.embed_model, embed_dim=dim, suspicious_articles=suspicious,
        )
        log(logger, logging.INFO, "index_full_done", **result)
        return result

    async def sync(self) -> dict[str, Any]:
        async with self._exclusive("sync"):
            state = await self._store.get_state()
            collection = await self._store.active_collection()
            if not collection or state.get("embed_model") != self._s.embed_model:
                log(logger, logging.INFO, "index_sync_requires_full")
                return await self._full()
            return await self._sync(collection, state)

    async def _sync(self, collection: str, state: dict[str, Any]) -> dict[str, Any]:
        t0 = time.monotonic()
        articles = await self._glpi.list_articles()
        existing = await self._store.article_index(collection)
        suspicious = {s["id"]: s for s in state.get("suspicious_articles") or []}
        updated = deleted = chunks = 0
        seen: set[int] = set()
        for a in articles:
            seen.add(a.id)
            h = self.article_hash(a)
            if existing.get(a.id, {}).get("content_hash") == h:
                continue
            r = await self._index_article(collection, a, h)
            updated += 1
            chunks += r["chunks"]
            if r["suspicious"]:
                suspicious[a.id] = {"id": a.id, "title": a.title}
            else:
                suspicious.pop(a.id, None)
        for article_id in set(existing) - seen:
            await self._store.delete_article(collection, article_id)
            suspicious.pop(article_id, None)
            deleted += 1
        result = {
            "mode": "sync", "articles_seen": len(articles), "articles_updated": updated,
            "chunks": chunks, "deleted": deleted, "suspicious": len(suspicious),
            "duration_ms": int((time.monotonic() - t0) * 1000),
        }
        await self._store.set_state(
            last_sync_at=_now_iso(), last_result=result, last_error=None,
            suspicious_articles=list(suspicious.values()),
        )
        log(logger, logging.INFO, "index_sync_done", **result)
        return result

    async def index_articles(self, article_ids: list[int]) -> dict[str, Any]:
        """Reindexa artigos específicos (ex.: aviso do plugin após editar/excluir um artigo)."""
        async with self._exclusive("articles"):
            collection = await self._store.active_collection()
            state = await self._store.get_state()
            if not collection or state.get("embed_model") != self._s.embed_model:
                return await self._full()
            suspicious = {s["id"]: s for s in state.get("suspicious_articles") or []}
            updated = deleted = 0
            for article_id in dict.fromkeys(int(i) for i in article_ids):
                article = await self._glpi.get_article(article_id)
                if article is None:
                    await self._store.delete_article(collection, article_id)
                    suspicious.pop(article_id, None)
                    deleted += 1
                    continue
                r = await self._index_article(collection, article, self.article_hash(article))
                updated += 1
                if r["suspicious"]:
                    suspicious[article.id] = {"id": article.id, "title": article.title}
                else:
                    suspicious.pop(article.id, None)
            result = {"mode": "articles", "articles_updated": updated, "deleted": deleted}
            await self._store.set_state(suspicious_articles=list(suspicious.values()))
            log(logger, logging.INFO, "index_articles_done", **result)
            return result

    async def record_error(self, mode: str, error: Exception) -> None:
        code = getattr(error, "code", type(error).__name__)
        log(logger, logging.ERROR, "index_failed", mode=mode, error=code, detail=str(error))
        try:
            await self._store.set_state(last_error={"mode": mode, "code": code, "at": _now_iso()})
        except Exception:
            pass
