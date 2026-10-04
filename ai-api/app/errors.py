"""Erros de domínio. Cada um vira um código estável na resposta HTTP, sem detalhes internos."""


class ServiceError(Exception):
    code = "internal_error"
    status = 500

    def __init__(self, detail: str = ""):
        super().__init__(detail or self.code)
        self.detail = detail


class LlmUnavailable(ServiceError):
    code = "llm_unavailable"
    status = 503


class LlmBusy(ServiceError):
    code = "llm_busy"
    status = 503


class VectorDbUnavailable(ServiceError):
    code = "vector_db_unavailable"
    status = 503


class GlpiUnavailable(ServiceError):
    code = "glpi_unavailable"
    status = 502


class GlpiNotFound(ServiceError):
    code = "glpi_not_found"
    status = 404


class GlpiAuthError(ServiceError):
    code = "glpi_auth_failed"
    status = 502


class IndexNotReady(ServiceError):
    code = "index_not_ready"
    status = 503


class SyncAlreadyRunning(ServiceError):
    code = "sync_running"
    status = 409
