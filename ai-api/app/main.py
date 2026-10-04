"""Aplicação FastAPI. Endpoints completos são implementados na Fase 8."""

from fastapi import FastAPI

app = FastAPI(title="Agente de Suporte N1 - ai-api", docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/api/live")
async def live() -> dict:
    return {"status": "ok"}
