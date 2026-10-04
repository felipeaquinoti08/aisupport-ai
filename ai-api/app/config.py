"""Configuração da ai-api, lida exclusivamente de variáveis de ambiente."""

from __future__ import annotations

import ipaddress
from functools import lru_cache

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Versão do contrato HTTP entre o plugin e a ai-api. O plugin recusa versões
# MAJOR diferentes da que ele conhece.
API_CONTRACT_VERSION = "1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    # --- ai-api ---------------------------------------------------------------
    ai_api_key: SecretStr = Field(min_length=32)
    ai_api_allowed_ips: str = ""
    max_body_bytes: int = 65536

    # --- GLPI (API REST v2, somente leitura da KB) ----------------------------
    glpi_url: str
    glpi_oauth_client_id: str = ""
    glpi_oauth_client_secret: SecretStr = SecretStr("")
    glpi_username: str = ""
    glpi_password: SecretStr = SecretStr("")
    glpi_verify_ssl: bool = True
    glpi_ca_bundle: str = ""
    glpi_timeout: float = 20.0
    glpi_page_size: int = 50
    # Fuso das datas devolvidas pelo GLPI (date_begin/date_end dos artigos)
    glpi_timezone: str = "America/Sao_Paulo"

    # --- Serviços internos ----------------------------------------------------
    ollama_url: str = "http://ollama:11434"
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: SecretStr = SecretStr("")
    qdrant_collection: str = "glpi_kb"

    # --- Modelos -------------------------------------------------------------
    llm_model: str = "qwen2.5:3b-instruct-q4_K_M"
    embed_model: str = "qwen3-embedding:0.6b"
    llm_num_ctx: int = 4096
    llm_temperature: float = 0.1
    llm_seed: int = 42
    llm_max_tokens: int = 512
    llm_timeout: float = 180.0
    embed_num_ctx: int = 512
    embed_timeout: float = 60.0
    ollama_keep_alive: str = "-1"
    # Instrução de consulta recomendada para modelos da família qwen3-embedding.
    # Vazio desativa (modelos que não usam instrução).
    embed_query_instruction: str = (
        "Given a user's IT support question, retrieve knowledge base passages that answer it"
    )
    llm_max_queue: int = 8

    # --- RAG -----------------------------------------------------------------
    rag_min_score: float = Field(default=0.55, ge=0.0, le=1.0)
    rag_min_term_coverage: float = Field(default=0.25, ge=0.0, le=1.0)
    rag_min_answer_overlap: float = Field(default=0.45, ge=0.0, le=1.0)
    rag_context_margin: float = Field(default=0.08, ge=0.0, le=1.0)
    rag_relative_margin: float = Field(default=0.15, ge=0.0, le=1.0)
    rag_top_k: int = Field(default=3, ge=1, le=10)
    rag_candidates: int = Field(default=20, ge=1, le=100)
    rag_max_chunks_per_article: int = Field(default=2, ge=1, le=10)
    rag_max_context_chars: int = Field(default=4500, ge=500, le=20000)
    rag_chunk_size: int = Field(default=1200, ge=200, le=8000)
    rag_chunk_overlap: int = Field(default=200, ge=0, le=2000)
    rag_exclude_suspicious: bool = True
    max_question_chars: int = 2000

    # --- Sincronização ---------------------------------------------------------
    sync_interval_minutes: int = Field(default=15, ge=0)

    # --- Logs ------------------------------------------------------------------
    log_level: str = "INFO"
    debug: bool = False

    @field_validator("glpi_url", "ollama_url", "qdrant_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("ai_api_allowed_ips")
    @classmethod
    def _validate_cidrs(cls, v: str) -> str:
        for item in filter(None, (p.strip() for p in v.split(","))):
            ipaddress.ip_network(item, strict=False)
        return v

    @property
    def allowed_networks(self) -> list[ipaddress.IPv4Network | ipaddress.IPv6Network]:
        return [
            ipaddress.ip_network(p.strip(), strict=False)
            for p in self.ai_api_allowed_ips.split(",")
            if p.strip()
        ]

    @property
    def glpi_verify(self) -> bool | str:
        if not self.glpi_verify_ssl:
            return False
        return self.glpi_ca_bundle or True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
