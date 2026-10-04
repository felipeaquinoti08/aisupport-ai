"""Montagem das dependências (um único ponto de criação dos clientes)."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, get_settings
from .glpi_client import GlpiClient
from .indexer.service import IndexerService
from .ollama import OllamaClient
from .rag import RagService
from .vectorstore import VectorStore


@dataclass
class Container:
    settings: Settings
    glpi: GlpiClient
    ollama: OllamaClient
    store: VectorStore
    indexer: IndexerService
    rag: RagService

    async def aclose(self) -> None:
        await self.glpi.aclose()
        await self.ollama.aclose()
        await self.store.aclose()


def build_container(settings: Settings | None = None, *, glpi=None, ollama=None, store=None) -> Container:
    settings = settings or get_settings()
    glpi = glpi or GlpiClient(settings)
    ollama = ollama or OllamaClient(settings)
    store = store or VectorStore(settings)
    return Container(
        settings=settings,
        glpi=glpi,
        ollama=ollama,
        store=store,
        indexer=IndexerService(settings, glpi, ollama, store),
        rag=RagService(settings, ollama, store),
    )
