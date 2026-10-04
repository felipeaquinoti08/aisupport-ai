"""Acesso ao Qdrant: coleções, trechos indexados, busca densa/esparsa e estado da sincronização.

A coleção é acessada por um alias (QDRANT_COLLECTION). A reindexação completa
cria uma coleção física nova e troca o alias de forma atômica no final, então
o chat continua respondendo durante a reindexação.
"""

from __future__ import annotations

import logging
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from .config import Settings
from .errors import IndexNotReady, VectorDbUnavailable

logger = logging.getLogger("vectorstore")

# O Qdrant fica na rede interna do compose; a chave em HTTP é esperada ali.
warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")

_NS = uuid.UUID("6f1c3c2e-8d1b-4b8e-9a51-2f0f6f2b9a10")
_FAR_FUTURE = 4102444800  # 2100-01-01
_STATE_POINT = 1


@dataclass
class Hit:
    chunk_id: str
    article_id: int
    chunk_index: int
    title: str
    text: str
    dense_score: float
    payload: dict[str, Any]


def point_id(article_id: int, chunk_index: int) -> str:
    return str(uuid.uuid5(_NS, f"{article_id}:{chunk_index}"))


def _wrap(fn):
    async def inner(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except (IndexNotReady, VectorDbUnavailable):
            raise
        except Exception as e:  # erros de rede/servidor do qdrant-client variam por versão
            raise VectorDbUnavailable(f"{fn.__name__}: {type(e).__name__}") from e
    inner.__name__ = fn.__name__
    return inner


class VectorStore:
    def __init__(self, settings: Settings, client: AsyncQdrantClient | None = None):
        self._s = settings
        self.alias = settings.qdrant_collection
        self.state_collection = f"{settings.qdrant_collection}_state"
        self._client = client or AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key.get_secret_value() or None,
            https=settings.qdrant_url.startswith("https"),
            timeout=15,
            prefer_grpc=False,
            check_compatibility=False,
        )

    async def aclose(self) -> None:
        await self._client.close()

    # --- Coleções ---------------------------------------------------------------

    @_wrap
    async def check(self) -> dict[str, Any]:
        t0 = time.monotonic()
        await self._client.get_collections()
        return {"latency_ms": int((time.monotonic() - t0) * 1000)}

    @_wrap
    async def active_collection(self) -> str | None:
        aliases = await self._client.get_aliases()
        for a in aliases.aliases:
            if a.alias_name == self.alias:
                return a.collection_name
        return None

    @_wrap
    async def create_collection(self, dim: int) -> str:
        name = f"{self.alias}_{int(time.time() * 1000)}"
        await self._client.create_collection(
            collection_name=name,
            vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE, on_disk=False)},
            sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            on_disk_payload=True,
        )
        for field, schema in (
            ("article_id", models.PayloadSchemaType.INTEGER),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
            ("valid_from", models.PayloadSchemaType.INTEGER),
            ("valid_until", models.PayloadSchemaType.INTEGER),
        ):
            await self._client.create_payload_index(name, field_name=field, field_schema=schema)
        return name

    @_wrap
    async def activate(self, collection: str) -> None:
        """Aponta o alias para `collection` e remove a coleção anterior."""
        previous = await self.active_collection()
        ops: list[Any] = []
        if previous:
            ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=self.alias)))
        ops.append(models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=collection, alias_name=self.alias)))
        await self._client.update_collection_aliases(change_aliases_operations=ops)
        if previous and previous != collection:
            await self._client.delete_collection(previous)

    @_wrap
    async def drop_collection(self, collection: str) -> None:
        await self._client.delete_collection(collection)

    @_wrap
    async def collection_dim(self, collection: str) -> int | None:
        info = await self._client.get_collection(collection)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict) and "dense" in vectors:
            return vectors["dense"].size
        return None

    # --- Escrita ------------------------------------------------------------------

    @_wrap
    async def upsert_article(self, collection: str, article_id: int, chunks: list[dict[str, Any]]) -> None:
        """Substitui todos os trechos de um artigo."""
        await self.delete_article(collection, article_id)
        points = [
            models.PointStruct(
                id=point_id(article_id, c["payload"]["chunk_index"]),
                vector={
                    "dense": c["dense"],
                    "sparse": models.SparseVector(indices=c["sparse"][0], values=c["sparse"][1]),
                },
                payload=c["payload"],
            )
            for c in chunks
        ]
        if points:
            await self._client.upsert(collection_name=collection, points=points, wait=True)

    @_wrap
    async def delete_article(self, collection: str, article_id: int) -> None:
        await self._client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="article_id", match=models.MatchValue(value=int(article_id)))])
            ),
            wait=True,
        )

    # --- Leitura --------------------------------------------------------------------

    @_wrap
    async def article_index(self, collection: str) -> dict[int, dict[str, Any]]:
        """Mapa article_id -> payload do primeiro trecho (hash, título, suspeita...)."""
        result: dict[int, dict[str, Any]] = {}
        offset = None
        flt = models.Filter(must=[models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0))])
        while True:
            points, offset = await self._client.scroll(
                collection_name=collection, scroll_filter=flt, limit=256, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for p in points:
                result[int(p.payload["article_id"])] = p.payload
            if offset is None:
                break
        return result

    @_wrap
    async def counts(self) -> dict[str, int]:
        collection = await self.active_collection()
        if not collection:
            return {"articles": 0, "chunks": 0}
        chunks = (await self._client.count(collection, exact=True)).count
        articles = (
            await self._client.count(
                collection,
                count_filter=models.Filter(must=[models.FieldCondition(key="chunk_index", match=models.MatchValue(value=0))]),
                exact=True,
            )
        ).count
        return {"articles": articles, "chunks": chunks}

    def _filter(self, allowed_ids: list[int] | None, now: int) -> models.Filter:
        must: list[Any] = [
            models.FieldCondition(key="valid_from", range=models.Range(lte=now)),
            models.FieldCondition(key="valid_until", range=models.Range(gt=now)),
        ]
        if self._s.rag_exclude_suspicious:
            must_not = [models.FieldCondition(key="suspicious", match=models.MatchValue(value=True))]
        else:
            must_not = []
        if allowed_ids is not None:
            must.append(models.FieldCondition(key="article_id", match=models.MatchAny(any=[int(i) for i in allowed_ids])))
        return models.Filter(must=must, must_not=must_not)

    async def _require_collection(self) -> str:
        collection = await self.active_collection()
        if not collection:
            raise IndexNotReady("a Base de Conhecimento ainda não foi indexada")
        return collection

    @_wrap
    async def search(
        self,
        dense: list[float],
        sparse: tuple[list[int], list[float]],
        limit: int,
        allowed_ids: list[int] | None = None,
    ) -> list[Hit]:
        """Busca híbrida. Retorna trechos ordenados por fusão RRF, cada um com seu score denso (cosseno)."""
        collection = await self._require_collection()
        if allowed_ids is not None and not allowed_ids:
            return []
        flt = self._filter(allowed_ids, int(time.time()))
        dense_res = await self._client.query_points(
            collection_name=collection, query=dense, using="dense", query_filter=flt,
            limit=limit, with_payload=True,
        )
        hits: dict[str, Hit] = {}
        ranks: dict[str, float] = {}
        for rank, p in enumerate(dense_res.points):
            hits[str(p.id)] = self._hit(p, float(p.score))
            ranks[str(p.id)] = 1.0 / (60 + rank)

        if sparse[0]:
            sparse_res = await self._client.query_points(
                collection_name=collection,
                query=models.SparseVector(indices=sparse[0], values=sparse[1]),
                using="sparse", query_filter=flt, limit=limit, with_payload=True, with_vectors=["dense"],
            )
            for rank, p in enumerate(sparse_res.points):
                key = str(p.id)
                if key not in hits:
                    vec = p.vector.get("dense") if isinstance(p.vector, dict) else None
                    score = sum(a * b for a, b in zip(dense, vec)) if vec else 0.0
                    hits[key] = self._hit(p, float(score))
                ranks[key] = ranks.get(key, 0.0) + 1.0 / (60 + rank)

        return sorted(hits.values(), key=lambda h: ranks[h.chunk_id], reverse=True)

    @staticmethod
    def _hit(p: Any, score: float) -> Hit:
        pl = p.payload or {}
        return Hit(
            chunk_id=str(p.id),
            article_id=int(pl.get("article_id", 0)),
            chunk_index=int(pl.get("chunk_index", 0)),
            title=str(pl.get("title", "")),
            text=str(pl.get("text", "")),
            dense_score=score,
            payload=pl,
        )

    # --- Estado da sincronização ------------------------------------------------------

    @_wrap
    async def get_state(self) -> dict[str, Any]:
        if not await self._client.collection_exists(self.state_collection):
            return {}
        points = await self._client.retrieve(self.state_collection, ids=[_STATE_POINT], with_payload=True)
        return dict(points[0].payload or {}) if points else {}

    @_wrap
    async def set_state(self, **values: Any) -> None:
        if not await self._client.collection_exists(self.state_collection):
            await self._client.create_collection(
                self.state_collection, vectors_config=models.VectorParams(size=1, distance=models.Distance.DOT)
            )
        state = await self.get_state()
        state.update(values)
        await self._client.upsert(
            self.state_collection,
            points=[models.PointStruct(id=_STATE_POINT, vector=[1.0], payload=state)],
            wait=True,
        )


def validity_window(date_begin: str | None, date_end: str | None, tz: str = "UTC") -> tuple[int, int]:
    """Converte date_begin/date_end do GLPI (hora local do servidor GLPI) em timestamps."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    zone = ZoneInfo(tz)

    def parse(value: str | None) -> int | None:
        if not value:
            return None
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=zone)
        return int(dt.timestamp())

    return parse(date_begin) or 0, parse(date_end) or _FAR_FUTURE
