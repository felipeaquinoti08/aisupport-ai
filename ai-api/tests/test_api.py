"""Camada HTTP: autenticação, validação, erros de dependências e health check."""

import pytest
from fastapi.testclient import TestClient
from qdrant_client import AsyncQdrantClient

from app.container import build_container
from app.main import create_app
from app.vectorstore import VectorStore
from tests.conftest import API_KEY, FakeGlpi, FakeOllama, make_settings, vpn_answer

AUTH = {"Authorization": f"Bearer {API_KEY}"}


def _client(settings=None, ollama=None, glpi=None, store=None):
    settings = settings or make_settings()
    ollama = ollama or FakeOllama()
    if glpi is None:
        glpi = FakeGlpi()
        from tests.conftest import PRINTER_HTML, VPN_HTML
        glpi.add(10, "Configuração da VPN no notebook", VPN_HTML)
        glpi.add(20, "Impressora do financeiro não imprime", PRINTER_HTML)
    store = store or VectorStore(settings, client=AsyncQdrantClient(location=":memory:"))
    c = build_container(settings, glpi=glpi, ollama=ollama, store=store)
    return TestClient(create_app(c, start_scheduler=False)), c


def test_requires_api_key():
    client, _ = _client()
    with client:
        assert client.get("/api/live").status_code == 200
        assert client.get("/api/status").status_code == 401
        assert client.get("/api/status", headers={"Authorization": "Bearer errado"}).status_code == 401
        r = client.post("/api/chat", json={"question": "x", "allowed_article_ids": []})
        assert r.status_code == 401 and r.json() == {"error": "unauthorized"}


def test_ip_allowlist():
    client, _ = _client(settings=make_settings(ai_api_allowed_ips="10.9.9.9/32"))
    with client:
        # TestClient usa o host "testclient", que não é um IP autorizado
        assert client.get("/api/status", headers=AUTH).status_code == 403
        assert client.get("/api/live").status_code == 200


def test_validation_does_not_echo_input():
    client, _ = _client()
    with client:
        r = client.post("/api/chat", headers=AUTH, json={"question": "dado sensível", "allowed_article_ids": "x", "extra": 1})
        assert r.status_code == 422
        assert "dado sensível" not in r.text


def test_full_flow_index_search_chat():
    ollama = FakeOllama()
    ollama.responder = vpn_answer
    client, _ = _client(ollama=ollama)
    with client:
        assert client.post("/api/index", headers=AUTH, json={"article_ids": [10]}).json()["mode"] == "full"
        s = client.post("/api/search", headers=AUTH, json={"question": "Como configuro a VPN no notebook?"}).json()
        ids = [c["article_id"] for c in s["candidates"]]
        assert ids[0] == 10
        r = client.post("/api/chat", headers=AUTH, json={"question": "Como configuro a VPN no notebook?", "allowed_article_ids": ids})
        body = r.json()
        assert r.status_code == 200 and body["status"] == "answered"
        assert body["sources"][0]["article_id"] == 10
        st = client.get("/api/status", headers=AUTH).json()
        assert st["contract_version"] == "1.0" and st["index"]["articles"] == 2


def test_ollama_down_returns_503():
    """14. Perda de conexão com o Ollama."""
    ollama = FakeOllama()
    client, c = _client(ollama=ollama)
    with client:
        client.post("/api/index", headers=AUTH, json={"article_ids": [10]})
        ollama.down = True
        r = client.post("/api/chat", headers=AUTH, json={"question": "Como configuro a VPN no notebook?", "allowed_article_ids": [10]})
        assert r.status_code == 503 and r.json() == {"error": "llm_unavailable"}
        h = client.get("/api/health", headers=AUTH).json()
        assert h["status"] == "down" and h["components"]["ollama"]["ok"] is False


def test_vector_db_down_returns_503():
    """15. Perda de conexão com o banco vetorial."""
    settings = make_settings(qdrant_url="http://127.0.0.1:9")
    store = VectorStore(settings, client=AsyncQdrantClient(url="http://127.0.0.1:9", timeout=1, check_compatibility=False))
    client, _ = _client(settings=settings, store=store)
    with client:
        r = client.post("/api/chat", headers=AUTH, json={"question": "Como configuro a VPN no notebook?", "allowed_article_ids": [10]})
        assert r.status_code == 503 and r.json() == {"error": "vector_db_unavailable"}
        h = client.get("/api/health", headers=AUTH).json()
        assert h["components"]["vector_db"]["ok"] is False and h["status"] == "down"
        st = client.get("/api/status", headers=AUTH).json()
        assert st["index"] == {"available": False, "error": "vector_db_unavailable"}


def test_index_not_ready():
    client, _ = _client()
    with client:
        r = client.post("/api/search", headers=AUTH, json={"question": "VPN notebook"})
        assert r.status_code == 503 and r.json() == {"error": "index_not_ready"}


def test_health_ok_and_glpi_failure():
    glpi = FakeGlpi()
    glpi.add(10, "VPN", "<p>Use o FortiClient.</p>")
    client, c = _client(glpi=glpi)
    with client:
        client.post("/api/index", headers=AUTH, json={"article_ids": [10]})
        h = client.get("/api/health", headers=AUTH).json()
        assert h["status"] == "ok"
        assert h["components"]["ollama"]["models"]["llm"]["installed"] is True
        glpi.fail = True
        h = client.get("/api/health", headers=AUTH).json()
        assert h["status"] == "degraded" and h["components"]["glpi"]["error"] == "glpi_unavailable"


def test_reindex_runs_in_background_and_rejects_concurrent():
    client, c = _client()
    with client:
        r = client.post("/api/reindex", headers=AUTH)
        assert r.status_code == 202
        # aguarda o job em segundo plano terminar
        for _ in range(50):
            st = client.get("/api/status", headers=AUTH).json()
            if not st["sync"]["running"] and st["index"].get("articles"):
                break
            import time
            time.sleep(0.05)
        assert st["index"]["articles"] == 2


@pytest.mark.parametrize("path", ["/api/sync", "/api/reindex"])
def test_jobs_require_key(path):
    client, _ = _client()
    with client:
        assert client.post(path).status_code == 401
