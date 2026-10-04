"""Cliente da API v2 do GLPI: token OAuth, paginação, renovação e falhas."""

import httpx
import pytest
import respx

from app.errors import GlpiAuthError, GlpiUnavailable
from app.glpi_client import GlpiClient
from tests.conftest import make_settings

pytestmark = pytest.mark.asyncio

BASE = "https://glpi.test"


def _article(i):
    return {"id": i, "name": f"Artigo {i}", "content": "<p>x</p>", "entity": {"id": 0, "name": "Raiz"},
            "categories": [{"id": 1, "name": "Rede"}], "is_recursive": True}


@respx.mock
async def test_token_and_pagination():
    token = respx.post(f"{BASE}/api.php/token").mock(return_value=httpx.Response(200, json={"access_token": "t1", "expires_in": 3600}))
    pages = respx.get(f"{BASE}/api.php/v2/Knowledgebase/Article").mock(side_effect=[
        httpx.Response(200, json=[_article(1), _article(2)], headers={"Content-Range": "0-1/3"}),
        httpx.Response(200, json=[_article(3)], headers={"Content-Range": "2-2/3"}),
    ])
    client = GlpiClient(make_settings(glpi_page_size=2))
    arts = await client.list_articles()
    await client.aclose()
    assert [a.id for a in arts] == [1, 2, 3]
    assert arts[0].categories == [{"id": 1, "name": "Rede"}] and arts[0].entity_id == 0
    assert token.call_count == 1
    assert pages.calls[0].request.headers["Authorization"] == "Bearer t1"
    sent = token.calls[0].request.content.decode()
    assert "grant_type=password" in sent and "scope=api" in sent


@respx.mock
async def test_expired_token_is_renewed_on_401():
    respx.post(f"{BASE}/api.php/token").mock(side_effect=[
        httpx.Response(200, json={"access_token": "old", "expires_in": 3600}),
        httpx.Response(200, json={"access_token": "new", "expires_in": 3600}),
    ])
    route = respx.get(f"{BASE}/api.php/v2/Knowledgebase/Article/5").mock(side_effect=[
        httpx.Response(401, json={}),
        httpx.Response(200, json=_article(5)),
    ])
    client = GlpiClient(make_settings())
    art = await client.get_article(5)
    await client.aclose()
    assert art.id == 5
    assert route.calls[1].request.headers["Authorization"] == "Bearer new"


@respx.mock
async def test_missing_article_returns_none():
    respx.post(f"{BASE}/api.php/token").mock(return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 3600}))
    respx.get(f"{BASE}/api.php/v2/Knowledgebase/Article/9").mock(return_value=httpx.Response(404, json={}))
    client = GlpiClient(make_settings())
    assert await client.get_article(9) is None
    await client.aclose()


@respx.mock
async def test_bad_credentials():
    respx.post(f"{BASE}/api.php/token").mock(return_value=httpx.Response(401, json={"error": "invalid_client"}))
    client = GlpiClient(make_settings())
    with pytest.raises(GlpiAuthError):
        await client.list_articles()
    await client.aclose()


@respx.mock
async def test_server_error_and_network_failure():
    respx.post(f"{BASE}/api.php/token").mock(return_value=httpx.Response(200, json={"access_token": "t", "expires_in": 3600}))
    respx.get(f"{BASE}/api.php/v2/Knowledgebase/Article").mock(return_value=httpx.Response(500))
    client = GlpiClient(make_settings())
    with pytest.raises(GlpiUnavailable):
        await client.list_articles()
    respx.get(f"{BASE}/api.php/v2/Knowledgebase/Article").mock(side_effect=httpx.ConnectError("boom"))
    with pytest.raises(GlpiUnavailable):
        await client.check()
    await client.aclose()


async def test_not_configured():
    client = GlpiClient(make_settings(glpi_password=""))
    assert client.configured is False
    with pytest.raises(GlpiAuthError):
        await client.check()
    await client.aclose()
