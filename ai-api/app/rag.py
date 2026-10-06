"""Orquestração do RAG: busca, decisão de evidência, LLM e validação da resposta."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

from . import sparse
from .config import Settings
from .guard import RagParams, evaluate_evidence, validate_answer, validate_external, validate_general
from .logging_setup import log
from .ollama import OllamaClient
from .prompts import (
    ANSWER_SCHEMA, GENERAL_SCHEMA, OUT_OF_SCOPE, SUMMARY_SCHEMA, UNKNOWN,
    build_chat_messages, build_external_messages, build_general_messages, build_summary_messages,
)
from .errors import LlmUnavailable
from .ollama import LlmResult
from .providers import ExternalLLM, PiiMasker, ProviderConfig, ProviderError, mask_messages, supports_web_search, unmask_json_text
from .text import clean_input, tokens
from .vectorstore import Hit, VectorStore

logger = logging.getLogger("rag")


@dataclass
class ChatOptions:
    """Ajustes enviados pelo plugin; sempre limitados a faixas seguras."""

    min_score: float | None = None
    top_k: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    model: str | None = None
    selected: bool = False
    instructions: str | None = None


@dataclass
class ChatResult:
    status: str  # answered | no_evidence | clarify
    reason: str
    answer: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)
    considered: list[dict[str, Any]] = field(default_factory=list)
    top_score: float = 0.0
    coverage: float = 0.0
    model: str = ""
    timings: dict[str, int] = field(default_factory=dict)
    # "local" ou "<provedor>:<modelo>"; fallback = motivo da queda para o modelo local
    provider: str = "local"
    fallback: str = ""


def _clamp(value, lo, hi, default):
    if value is None:
        return default
    return max(lo, min(hi, value))


class _EmbeddingCache:
    """Evita recalcular o embedding da mesma pergunta entre /search e /chat."""

    def __init__(self, size: int = 64, ttl: float = 300.0):
        self._data: OrderedDict[str, tuple[float, list[float]]] = OrderedDict()
        self._size = size
        self._ttl = ttl

    def get(self, key: str) -> list[float] | None:
        item = self._data.get(key)
        if not item or time.monotonic() - item[0] > self._ttl:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return item[1]

    def put(self, key: str, vec: list[float]) -> None:
        self._data[key] = (time.monotonic(), vec)
        self._data.move_to_end(key)
        while len(self._data) > self._size:
            self._data.popitem(last=False)


class RagService:
    def __init__(self, settings: Settings, ollama: OllamaClient, store: VectorStore, external: ExternalLLM | None = None):
        self._s = settings
        self._ollama = ollama
        self._store = store
        self._external = external or ExternalLLM()
        self._cache = _EmbeddingCache()
        self._models_cache: tuple[float, list[str]] = (0.0, [])

    # --- Parâmetros -----------------------------------------------------------------

    def params(self, opts: ChatOptions | None = None) -> RagParams:
        s = self._s
        opts = opts or ChatOptions()
        return RagParams(
            min_score=_clamp(opts.min_score, 0.2, 0.95, s.rag_min_score),
            min_term_coverage=s.rag_min_term_coverage,
            min_answer_overlap=s.rag_min_answer_overlap,
            context_margin=s.rag_context_margin,
            relative_margin=s.rag_relative_margin,
            top_k=int(_clamp(opts.top_k, 1, 6, s.rag_top_k)),
            max_chunks_per_article=s.rag_max_chunks_per_article,
            max_context_chars=s.rag_max_context_chars,
        )

    async def _resolve_model(self, requested: str | None) -> str:
        if not requested or requested == self._s.llm_model:
            return self._s.llm_model
        ts, models = self._models_cache
        if time.monotonic() - ts > 60:
            try:
                models = await self._ollama.list_models()
            except Exception:
                models = []
            self._models_cache = (time.monotonic(), models)
        return requested if requested in models else self._s.llm_model

    # --- Geração: provedor externo ou modelo local ----------------------------------------

    async def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        schema: dict[str, Any] | None,
        max_tokens: int,
        temperature: float | None,
        model: str | None,
        provider: ProviderConfig | None,
    ) -> tuple[LlmResult, str, str]:
        """Resultado, quem gerou ("local" ou "<provedor>:<modelo>") e o motivo da
        queda para o modelo local. Com provedor externo, os dados pessoais são
        mascarados antes do envio e restaurados na resposta."""
        fallback = ""
        if provider is not None:
            masker = PiiMasker() if provider.mask_pii else None
            outgoing = mask_messages(messages, masker) if masker else messages
            try:
                # Modelos de raciocínio contam o raciocínio no limite de saída
                result = await self._external.chat(
                    provider, outgoing, schema=schema, max_tokens=max(max_tokens, 4000), temperature=temperature,
                )
            except ProviderError as e:
                log(logger, logging.WARNING, "provider_failed", provider=provider.kind, model=provider.model,
                    error=e.code, status=e.status, fallback_local=provider.fallback_local)
                if not provider.fallback_local:
                    raise LlmUnavailable(f"provider:{e.code}") from e
                fallback = e.code
            else:
                if masker and masker.count:
                    result.text = unmask_json_text(result.text, masker) if schema else masker.unmask(result.text)
                return result, provider.label, ""
        result = await self._ollama.chat(
            messages,
            model=await self._resolve_model(model),
            temperature=temperature,
            max_tokens=max_tokens,
            json_output=schema if schema else True,
        )
        return result, "local", fallback

    # --- Busca -------------------------------------------------------------------------

    @staticmethod
    def retrieval_query(question: str, history: list[dict[str, str]] | None) -> str:
        """Perguntas curtas de continuação ("e no celular?") herdam o contexto da anterior."""
        if history and len(tokens(question)) < 5:
            previous = [m.get("content", "") for m in history if m.get("role") == "user"]
            if previous:
                return f"{previous[-1]}\n{question}"
        return question

    async def _embed_query(self, text: str) -> list[float]:
        key = hashlib.sha256(f"{self._s.embed_model}\0{text}".encode()).hexdigest()
        vec = self._cache.get(key)
        if vec is None:
            vec = (await self._ollama.embed([text], is_query=True))[0]
            self._cache.put(key, vec)
        return vec

    async def retrieve(self, query: str, limit: int, allowed_ids: list[int] | None = None) -> list[Hit]:
        dense = await self._embed_query(query)
        return await self._store.search(dense, sparse.query_vector(query), limit, allowed_ids)

    async def search(self, question: str, history: list[dict[str, str]] | None = None, limit: int | None = None) -> dict[str, Any]:
        """Candidatos por artigo (sem conteúdo) para o plugin checar permissões."""
        t0 = time.monotonic()
        question = clean_input(question)[: self._s.max_question_chars]
        if not tokens(question):
            return {"candidates": [], "timings": {"total_ms": 0}}
        hits = await self.retrieve(self.retrieval_query(question, history), limit or self._s.rag_candidates)
        best: dict[int, dict[str, Any]] = {}
        for h in hits:
            cur = best.get(h.article_id)
            if cur is None or h.dense_score > cur["score"]:
                best[h.article_id] = {
                    "article_id": h.article_id,
                    "title": h.title,
                    "score": round(h.dense_score, 4),
                    "url": h.payload.get("url", ""),
                    "categories": list(h.payload.get("categories") or []),
                }
        candidates = sorted(best.values(), key=lambda c: c["score"], reverse=True)
        return {"candidates": candidates, "timings": {"total_ms": int((time.monotonic() - t0) * 1000)}}

    # --- Chat ----------------------------------------------------------------------------

    async def chat(
        self,
        question: str,
        allowed_ids: list[int],
        history: list[dict[str, str]] | None = None,
        opts: ChatOptions | None = None,
        provider: ProviderConfig | None = None,
    ) -> ChatResult:
        t0 = time.monotonic()
        opts = opts or ChatOptions()
        p = self.params(opts)
        question = clean_input(question)[: self._s.max_question_chars]

        if not tokens(question):
            return ChatResult(status="clarify", reason="empty_query", timings={"total_ms": 0})

        query = self.retrieval_query(question, history)
        hits = await self.retrieve(query, self._s.rag_candidates, allowed_ids)
        t_retrieval = int((time.monotonic() - t0) * 1000)
        selected = bool(opts.selected) and len(allowed_ids) == 1
        decision = evaluate_evidence(question, hits, p, alt_question=query, trusted=selected)
        considered = self._considered(hits)

        if not decision.ok:
            log(logger, logging.INFO, "chat_no_evidence", reason=decision.reason,
                top_score=round(decision.top_score, 4), coverage=round(decision.coverage, 3), hits=len(hits))
            return ChatResult(
                status="no_evidence", reason=decision.reason, considered=considered,
                top_score=decision.top_score, coverage=decision.coverage,
                timings={"retrieval_ms": t_retrieval, "total_ms": t_retrieval},
            )

        documents = [{"article_id": h.article_id, "title": h.title, "text": h.text} for h in decision.context]
        result, used, fallback = await self._generate(
            build_chat_messages(
                question if not selected or not documents
                else f"{question}\n(O usuário indicou que o problema é sobre o assunto do documento \"{documents[0]['title']}\".)",
                documents,
                opts.instructions,
            ),
            schema=ANSWER_SCHEMA,
            model=opts.model,
            temperature=_clamp(opts.temperature, 0.0, 1.0, None),
            max_tokens=int(_clamp(opts.max_tokens, 64, 1024, self._s.llm_max_tokens)),
            provider=provider,
        )
        check = validate_answer(result.text, decision.context, p)
        timings = {
            "retrieval_ms": t_retrieval,
            "llm_ms": result.duration_ms,
            "total_ms": int((time.monotonic() - t0) * 1000),
        }
        log(logger, logging.INFO, "chat_answer_checked", ok=check.ok, reason=check.reason,
            top_score=round(decision.top_score, 4), overlap=round(check.overlap, 3), provider=used, fallback=fallback,
            prompt_tokens=result.prompt_tokens, output_tokens=result.output_tokens, **timings)
        if self._s.debug:
            log(logger, logging.DEBUG, "chat_debug", question=question, answer=result.text)

        if not check.ok:
            return ChatResult(
                status="no_evidence", reason=check.reason, considered=considered,
                top_score=decision.top_score, coverage=decision.coverage, model=result.model, timings=timings,
                provider=used, fallback=fallback,
            )

        by_article = {h.article_id: h for h in decision.context}
        sources = []
        for aid in check.source_ids:
            h = by_article[aid]
            best = max(x.dense_score for x in decision.context if x.article_id == aid)
            sources.append({"article_id": aid, "title": h.title, "score": round(best, 4), "url": h.payload.get("url", "")})
        return ChatResult(
            status="answered", reason="ok", answer=check.answer, sources=sources, considered=considered,
            top_score=decision.top_score, coverage=decision.coverage, model=result.model, timings=timings,
            provider=used, fallback=fallback,
        )

    @staticmethod
    def _considered(hits: list[Hit]) -> list[dict[str, Any]]:
        seen: dict[int, dict[str, Any]] = {}
        for h in hits:
            if h.article_id not in seen or h.dense_score > seen[h.article_id]["score"]:
                seen[h.article_id] = {"article_id": h.article_id, "title": h.title, "score": round(h.dense_score, 4)}
        return sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:5]

    # --- Resumo para o chamado -------------------------------------------------------------

    async def summarize(
        self,
        question: str,
        transcript: list[dict[str, str]],
        instructions: str | None = None,
        provider: ProviderConfig | None = None,
    ) -> dict[str, Any]:
        question = clean_input(question)[: self._s.max_question_chars]
        transcript = [
            {"role": m.get("role", "user"), "content": clean_input(m.get("content", ""))[:1500]}
            for m in transcript[-12:]
        ]
        fallback_title = question.splitlines()[0][:80] if question else "Solicitação via Agente N1"
        try:
            result, used, _ = await self._generate(
                build_summary_messages(question, transcript, instructions),
                schema=SUMMARY_SCHEMA, max_tokens=self._s.summary_max_tokens, temperature=0.0, model=None, provider=provider,
            )
            data = json.loads(result.text)
            title = clean_input(str(data.get("titulo") or ""))[:80]
            summary = clean_input(str(data.get("resumo") or ""))[:600]
            if title and summary:
                return {"title": title, "summary": summary, "generated": True, "provider": used}
        except Exception as e:
            log(logger, logging.WARNING, "summary_fallback", error=type(e).__name__)
        return {"title": fallback_title, "summary": question[:600], "generated": False, "provider": ""}

    # --- Orientação geral (fora da KB) ------------------------------------------------------

    async def general(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        instructions: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        provider: ProviderConfig | None = None,
    ) -> dict[str, Any]:
        """Só é chamada pelo plugin quando a KB não resolveu e o administrador
        liberou respostas fora dela; o plugin mostra o aviso ao usuário."""
        t0 = time.monotonic()
        question = clean_input(question)[: self._s.max_question_chars]
        if not tokens(question):
            return {"status": "declined", "reason": "empty_query", "answer": "", "model": "", "timings": {"total_ms": 0}}
        history = [
            {"role": m.get("role", "user"), "content": clean_input(m.get("content", ""))[:1500]}
            for m in (history or [])[-6:]
        ]
        result, used, fallback = await self._generate(
            build_general_messages(question, history, instructions),
            schema=GENERAL_SCHEMA,
            model=model,
            temperature=_clamp(temperature, 0.0, 1.0, None),
            max_tokens=int(_clamp(max_tokens, 64, 1024, self._s.llm_max_tokens)),
            provider=provider,
        )
        check = validate_general(result.text)
        timings = {"llm_ms": result.duration_ms, "total_ms": int((time.monotonic() - t0) * 1000)}
        log(logger, logging.INFO, "general_answer_checked", ok=check.ok, reason=check.reason, provider=used, fallback=fallback,
            prompt_tokens=result.prompt_tokens, output_tokens=result.output_tokens, **timings)
        return {
            "status": "answered" if check.ok else "declined",
            "reason": check.reason,
            "answer": check.answer,
            "model": result.model,
            "timings": timings,
            "provider": used,
            "fallback": fallback,
        }

    # --- Escopo + fonte externa (exige provedor) ----------------------------------------------

    async def external(
        self,
        question: str,
        provider: ProviderConfig,
        scope: dict[str, Any],
        allowed_ids: list[int] | None = None,
        history: list[dict[str, str]] | None = None,
        opts: ChatOptions | None = None,
        include_kb: bool = False,
    ) -> dict[str, Any]:
        """Resposta dentro do escopo definido pela empresa, a partir dos artigos
        do GLPI (include_kb) e/ou da fonte externa: pesquisa ao vivo nos sites
        permitidos (OpenAI/Anthropic) ou o conhecimento do próprio modelo."""
        t0 = time.monotonic()
        opts = opts or ChatOptions()
        p = self.params(opts)
        question = clean_input(question)[: self._s.max_question_chars]
        history = [
            {"role": m.get("role", "user"), "content": clean_input(m.get("content", ""))[:1500]}
            for m in (history or [])[-6:]
        ]
        empty = {"answer": "", "sources": [], "web_sources": [], "considered": [], "top_score": 0.0,
                 "model": "", "provider": provider.label, "fallback": "", "searched": False}
        if not tokens(question):
            return {**empty, "status": "declined", "reason": "empty_query", "timings": {"total_ms": 0}}

        context, considered, top = [], [], 0.0
        if include_kb and allowed_ids:
            query = self.retrieval_query(question, history)
            hits = await self.retrieve(query, self._s.rag_candidates, allowed_ids)
            decision = evaluate_evidence(question, hits, p, alt_question=query)
            context = decision.context if decision.ok else []
            considered, top = self._considered(hits), decision.top_score
        t_retrieval = int((time.monotonic() - t0) * 1000)

        domains = list(scope.get("domains") or [])
        search = bool(scope.get("web_search", True)) and bool(domains) and supports_web_search(provider)
        messages = build_external_messages(
            question, history, [{"article_id": h.article_id, "title": h.title, "text": h.text} for h in context],
            str(scope.get("text") or ""), domains, search, opts.instructions,
        )
        masker = PiiMasker() if provider.mask_pii else None
        outgoing = mask_messages(messages, masker) if masker else messages
        try:
            res = await self._external.answer_text(
                provider, outgoing, domains=domains if search else None,
                max_tokens=8000 if provider.kind == "anthropic" else 4000,
                temperature=_clamp(opts.temperature, 0.0, 1.0, None),
            )
        except ProviderError as e:
            log(logger, logging.WARNING, "provider_failed", provider=provider.kind, model=provider.model,
                error=e.code, status=e.status, fallback_local=provider.fallback_local, mode="external")
            if include_kb and provider.fallback_local:
                # Só a parte da KB pode ser feita pelo modelo local
                local = await self.chat(question, allowed_ids or [], history, opts)
                return {**asdict(local), "web_sources": [], "searched": False, "fallback": e.code}
            return {**empty, "status": "declined", "reason": "provider_failed", "fallback": e.code,
                    "considered": considered, "timings": {"retrieval_ms": t_retrieval, "total_ms": int((time.monotonic() - t0) * 1000)}}

        text = masker.unmask(res.text) if masker else res.text
        check = validate_external(text, context, domains, res.citations, OUT_OF_SCOPE, UNKNOWN)
        timings = {"retrieval_ms": t_retrieval, "llm_ms": res.duration_ms, "total_ms": int((time.monotonic() - t0) * 1000)}
        log(logger, logging.INFO, "external_answer_checked", ok=check.ok, reason=check.reason, provider=res.model,
            searched=res.searched, kb=len(check.kb_ids), web=len(check.web_sources), **timings)
        by_article = {h.article_id: h for h in context}
        sources = []
        for aid in check.kb_ids:
            h = by_article[aid]
            best = max(x.dense_score for x in context if x.article_id == aid)
            sources.append({"article_id": aid, "title": h.title, "score": round(best, 4), "url": h.payload.get("url", "")})
        return {
            "status": "answered" if check.ok else "declined",
            "reason": check.reason,
            "answer": check.answer,
            "sources": sources,
            "web_sources": check.web_sources,
            "considered": considered,
            "top_score": top,
            "model": res.model,
            "provider": res.model,
            "fallback": "",
            "searched": res.searched,
            "timings": timings,
        }

    async def test_provider(self, provider: ProviderConfig) -> dict[str, Any]:
        """Teste do provedor externo (botão do plugin): sem queda para o modelo local."""
        try:
            result = await self._external.chat(
                provider,
                [{"role": "system", "content": "Responda em JSON."},
                 {"role": "user", "content": 'Responda {"ok": true}.'}],
                schema={"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
                max_tokens=2000,
                temperature=0.0,
            )
        except ProviderError as e:
            return {"ok": False, "error": e.code, "status": e.status, "detail": e.detail}
        try:
            ok = json.loads(result.text).get("ok") is True
        except (ValueError, AttributeError):
            ok = False
        return {"ok": ok, "error": "" if ok else "invalid_response", "model": result.model,
                "duration_ms": result.duration_ms, "prompt_tokens": result.prompt_tokens, "output_tokens": result.output_tokens}

    async def test_llm(self) -> dict[str, Any]:
        result = await self._ollama.chat(
            [{"role": "user", "content": "Responda apenas com a palavra OK."}], max_tokens=8, temperature=0.0
        )
        return {
            "model": result.model, "output": result.text[:20], "duration_ms": result.duration_ms,
            "prompt_tokens": result.prompt_tokens, "output_tokens": result.output_tokens,
        }
