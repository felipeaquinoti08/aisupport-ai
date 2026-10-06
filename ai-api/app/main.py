"""API HTTP da ai-api (FastAPI). Acesso exclusivo do plugin do GLPI, por chave compartilhada."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import API_CONTRACT_VERSION, Settings
from .container import Container, build_container
from .errors import ServiceError, SyncAlreadyRunning
from .logging_setup import log, setup_logging
from .rag import ChatOptions
from .schemas import ChatRequest, GeneralRequest, IndexRequest, ProviderTestRequest, SearchRequest, SummarizeRequest

APP_VERSION = "1.0.0"
logger = logging.getLogger("api")


class Unauthorized(ServiceError):
    code = "unauthorized"
    status = 401


class NotConfigured(ServiceError):
    code = "not_configured"
    status = 500


def _secrets(s: Settings) -> list[str]:
    return [
        s.ai_api_key.get_secret_value(),
        s.qdrant_api_key.get_secret_value(),
        s.glpi_oauth_client_secret.get_secret_value(),
        s.glpi_password.get_secret_value(),
    ]


async def _scheduler(c: Container) -> None:
    """Sincronização periódica da KB (a primeira execução indexa tudo se o índice estiver vazio)."""
    await asyncio.sleep(20)
    interval = c.settings.sync_interval_minutes * 60
    while True:
        try:
            if c.glpi.configured:
                await c.indexer.sync()
        except SyncAlreadyRunning:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await c.indexer.record_error("sync", e)
        await asyncio.sleep(interval)


def require_key(request: Request) -> Container:
    c: Container = request.app.state.c
    header = request.headers.get("authorization", "")
    token = header[7:] if header.lower().startswith("bearer ") else ""
    expected = c.settings.ai_api_key.get_secret_value()
    if not token or not hmac.compare_digest(token.encode(), expected.encode()):
        raise Unauthorized()
    return c


def create_app(container: Container | None = None, *, start_scheduler: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = container or build_container()
        setup_logging(c.settings.log_level, _secrets(c.settings))
        app.state.c = c
        app.state.jobs = set()
        task = None
        if start_scheduler and c.settings.sync_interval_minutes > 0:
            task = asyncio.create_task(_scheduler(c))
        log(logger, logging.INFO, "startup", version=APP_VERSION, llm=c.settings.llm_model, embed=c.settings.embed_model)
        try:
            yield
        finally:
            if task:
                task.cancel()
            for job in list(app.state.jobs):
                job.cancel()
            await c.aclose()

    app = FastAPI(
        title="Agente de Suporte N1 - ai-api",
        version=APP_VERSION,
        docs_url=None, redoc_url=None, openapi_url=None,
        lifespan=lifespan,
    )

    # --- Middlewares e erros ---------------------------------------------------------

    @app.middleware("http")
    async def guard(request: Request, call_next):
        s: Settings = request.app.state.c.settings
        client_ip = request.client.host if request.client else ""
        if request.url.path != "/api/live" and s.allowed_networks:
            try:
                ip = ipaddress.ip_address(client_ip)
                allowed = ip.is_loopback or any(ip in net for net in s.allowed_networks)
            except ValueError:
                allowed = False
            if not allowed:
                log(logger, logging.WARNING, "ip_denied", ip=client_ip, path=request.url.path)
                return JSONResponse({"error": "forbidden"}, status_code=403)
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > s.max_body_bytes):
            return JSONResponse({"error": "payload_too_large"}, status_code=413)
        t0 = time.monotonic()
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path != "/api/live":
            log(logger, logging.INFO, "request", method=request.method, path=request.url.path,
                status=response.status_code, ms=int((time.monotonic() - t0) * 1000))
        return response

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        level = logging.INFO if isinstance(exc, Unauthorized) else logging.WARNING
        log(logger, level, "service_error", path=request.url.path, error=exc.code, detail=exc.detail)
        headers = {"Retry-After": "10"} if exc.status == 503 else None
        return JSONResponse({"error": exc.code}, status_code=exc.status, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Não devolve os valores recebidos (podem conter dados pessoais).
        fields = sorted({".".join(str(p) for p in e.get("loc", [])[1:]) for e in exc.errors()})
        return JSONResponse({"error": "invalid_request", "fields": fields}, status_code=422)

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, exc: Exception):
        logger.exception("unexpected_error", extra={"data": {"path": request.url.path}})
        return JSONResponse({"error": "internal_error"}, status_code=500)

    # --- Endpoints -----------------------------------------------------------------------

    @app.get("/api/live")
    async def live() -> dict:
        return {"status": "ok"}

    @app.get("/api/health")
    async def health(c: Container = Depends(require_key)) -> dict:
        return await _health(c)

    @app.get("/api/status")
    async def status(c: Container = Depends(require_key)) -> dict:
        s = c.settings
        try:
            index: dict[str, Any] = {**(await c.store.counts()), **(await c.store.get_state()), "available": True}
        except ServiceError as e:
            index = {"available": False, "error": e.code}
        return {
            "contract_version": API_CONTRACT_VERSION,
            "version": APP_VERSION,
            "models": {"llm": s.llm_model, "embeddings": s.embed_model},
            "rag": {
                "min_score": s.rag_min_score, "top_k": s.rag_top_k, "candidates": s.rag_candidates,
                "min_term_coverage": s.rag_min_term_coverage, "min_answer_overlap": s.rag_min_answer_overlap,
            },
            "index": index,
            "sync": {"running": c.indexer.running, "interval_minutes": s.sync_interval_minutes},
            "llm_queue": c.ollama.queue_size,
        }

    @app.post("/api/search")
    async def search(body: SearchRequest, c: Container = Depends(require_key)) -> dict:
        return await c.rag.search(body.question, [m.model_dump() for m in body.history], body.limit)

    @app.post("/api/chat")
    async def chat(body: ChatRequest, c: Container = Depends(require_key)) -> dict:
        result = await c.rag.chat(
            body.question,
            sorted({i for i in body.allowed_article_ids if i > 0}),
            [m.model_dump() for m in body.history],
            ChatOptions(**body.options.model_dump()),
            body.provider.config() if body.provider else None,
        )
        return asdict(result)

    @app.post("/api/summarize")
    async def summarize(body: SummarizeRequest, c: Container = Depends(require_key)) -> dict:
        return await c.rag.summarize(
            body.question, [m.model_dump() for m in body.transcript], body.instructions,
            body.provider.config() if body.provider else None,
        )

    @app.post("/api/general")
    async def general(body: GeneralRequest, c: Container = Depends(require_key)) -> dict:
        return await c.rag.general(
            body.question, [m.model_dump() for m in body.history], body.instructions,
            body.temperature, body.max_tokens, body.model,
            body.provider.config() if body.provider else None,
        )

    @app.post("/api/provider-test")
    async def provider_test(body: ProviderTestRequest, c: Container = Depends(require_key)) -> dict:
        return await c.rag.test_provider(body.provider.config())

    @app.post("/api/test-llm")
    async def test_llm(c: Container = Depends(require_key)) -> dict:
        return await c.rag.test_llm()

    @app.post("/api/index")
    async def index_articles(body: IndexRequest, c: Container = Depends(require_key)) -> dict:
        return await c.indexer.index_articles(body.ids())

    def _start_job(request: Request, c: Container, mode: str) -> JSONResponse:
        if c.indexer.running:
            raise SyncAlreadyRunning(c.indexer.running)

        async def job():
            try:
                await (c.indexer.full() if mode == "full" else c.indexer.sync())
            except SyncAlreadyRunning:
                pass
            except Exception as e:
                await c.indexer.record_error(mode, e)

        task = asyncio.create_task(job())
        request.app.state.jobs.add(task)
        task.add_done_callback(request.app.state.jobs.discard)
        return JSONResponse({"status": "started", "mode": mode}, status_code=202)

    @app.post("/api/sync")
    async def sync(request: Request, c: Container = Depends(require_key)):
        return _start_job(request, c, "sync")

    @app.post("/api/reindex")
    async def reindex(request: Request, c: Container = Depends(require_key)):
        return _start_job(request, c, "full")

    return app


async def _timed(coro) -> dict[str, Any]:
    t0 = time.monotonic()
    try:
        extra = await coro
        return {"ok": True, "latency_ms": int((time.monotonic() - t0) * 1000), **(extra or {})}
    except ServiceError as e:
        return {"ok": False, "error": e.code, "latency_ms": int((time.monotonic() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "latency_ms": int((time.monotonic() - t0) * 1000)}


async def _health(c: Container) -> dict[str, Any]:
    s = c.settings

    async def ollama_check():
        installed = await c.ollama.list_models()
        loaded = await c.ollama.loaded_models()
        return {"models": {
            "llm": {"name": s.llm_model, "installed": s.llm_model in installed, "loaded": s.llm_model in loaded},
            "embeddings": {"name": s.embed_model, "installed": s.embed_model in installed, "loaded": s.embed_model in loaded},
        }}

    async def vector_check():
        await c.store.check()
        return {"collection": bool(await c.store.active_collection())}

    async def glpi_check():
        if not c.glpi.configured:
            raise NotConfigured()
        await c.glpi.check()
        return {}

    ollama, vector_db, glpi = await asyncio.gather(_timed(ollama_check()), _timed(vector_check()), _timed(glpi_check()))
    models_ok = bool(ollama["ok"]) and all(m["installed"] for m in ollama.get("models", {}).values())
    core_ok = bool(vector_db["ok"]) and models_ok
    overall = "ok" if core_ok and glpi["ok"] and vector_db.get("collection") else ("degraded" if core_ok else "down")
    return {
        "status": overall,
        "contract_version": API_CONTRACT_VERSION,
        "components": {"api": {"ok": True, "version": APP_VERSION}, "ollama": ollama, "vector_db": vector_db, "glpi": glpi},
    }


app = create_app()
