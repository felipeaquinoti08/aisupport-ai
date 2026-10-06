"""Contratos HTTP entre o plugin e a ai-api (versão em config.API_CONTRACT_VERSION)."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MAX_ID = 2**31 - 1


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class ProviderIn(BaseModel):
    """Provedor externo que redige a resposta (a busca continua local).
    Enviado pelo plugin a cada requisição; a chave não é gravada nem logada."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["openai", "azure", "compatible", "anthropic"]
    api_key: str = Field(min_length=1, max_length=500, repr=False)
    model: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9._:/@-]+$")
    base_url: str = Field(default="", max_length=500, pattern=r"^(https?://[^\s]+)?$")
    api_version: str = Field(default="", max_length=40, pattern=r"^[A-Za-z0-9._-]*$")
    effort: Literal["", "low", "medium", "high"] = ""
    timeout: float = Field(default=45, ge=5, le=180)
    mask_pii: bool = True
    fallback_local: bool = True

    def config(self):
        from .providers import ProviderConfig
        return ProviderConfig(**self.model_dump())


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)
    limit: int | None = Field(default=None, ge=1, le=100)


class ChatOptionsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_score: float | None = Field(default=None, ge=0, le=1)
    top_k: int | None = Field(default=None, ge=1, le=10)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    model: str | None = Field(default=None, max_length=120, pattern=r"^[A-Za-z0-9._:/-]+$")
    # The user picked this article among the suggestions (allowed_article_ids
    # must then hold exactly that article): the relevance gates are skipped,
    # the answer must still come from the document.
    selected: bool = False
    # Instruções do administrador (tom, formato). Entram no prompt abaixo das
    # regras fixas e nunca liberam respostas fora dos documentos.
    instructions: str | None = Field(default=None, max_length=1500)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    # IDs de artigos que o plugin confirmou que o usuário pode ver (canViewItem).
    allowed_article_ids: list[int] = Field(max_length=200)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)
    options: ChatOptionsIn = Field(default_factory=ChatOptionsIn)
    provider: ProviderIn | None = None


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    transcript: list[HistoryMessage] = Field(default_factory=list, max_length=40)
    instructions: str | None = Field(default=None, max_length=1000)
    provider: ProviderIn | None = None


class GeneralRequest(BaseModel):
    """Orientação geral de suporte, usada pelo plugin só quando a KB não resolve
    e o administrador liberou respostas fora da Base de Conhecimento."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)
    instructions: str | None = Field(default=None, max_length=1500)
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    model: str | None = Field(default=None, max_length=120, pattern=r"^[A-Za-z0-9._:/-]+$")
    provider: ProviderIn | None = None


class ScopeIn(BaseModel):
    """O que o agente pode responder fora da KB do GLPI e em quais sites se basear."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(default="", max_length=1500)
    domains: list[str] = Field(default_factory=list, max_length=20)
    web_search: bool = True

    @field_validator("domains")
    @classmethod
    def _domains(cls, value: list[str]) -> list[str]:
        out = []
        for d in value:
            d = d.strip().lower().removeprefix("https://").removeprefix("http://").split("/")[0].removeprefix("www.")
            if not re.fullmatch(r"(?:[a-z0-9-]+\.)+[a-z]{2,}", d):
                raise ValueError("domínio inválido")
            if d not in out:
                out.append(d)
        return out


class ExternalRequest(BaseModel):
    """Resposta com escopo e fonte externa (exige provedor). include_kb=True
    combina os artigos do GLPI que o usuário pode ver; False = só fonte externa."""

    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    allowed_article_ids: list[int] = Field(default_factory=list, max_length=200)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)
    options: ChatOptionsIn = Field(default_factory=ChatOptionsIn)
    provider: ProviderIn
    scope: ScopeIn = Field(default_factory=ScopeIn)
    include_kb: bool = False


class ProviderTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: ProviderIn


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    article_ids: list[int] = Field(min_length=1, max_length=200)

    def ids(self) -> list[int]:
        return [i for i in self.article_ids if 0 < i <= _MAX_ID]
