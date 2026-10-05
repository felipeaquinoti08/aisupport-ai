"""Contratos HTTP entre o plugin e a ai-api (versão em config.API_CONTRACT_VERSION)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

_MAX_ID = 2**31 - 1


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


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


class SummarizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    transcript: list[HistoryMessage] = Field(default_factory=list, max_length=40)
    instructions: str | None = Field(default=None, max_length=1000)


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


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    article_ids: list[int] = Field(min_length=1, max_length=200)

    def ids(self) -> list[int]:
        return [i for i in self.article_ids if 0 < i <= _MAX_ID]
