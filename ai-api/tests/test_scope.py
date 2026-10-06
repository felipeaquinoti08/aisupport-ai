"""Escopo + fonte externa: artigos do GLPI, pesquisa nos sites permitidos e
conhecimento do modelo, sempre com links restritos aos sites liberados."""

import json
from types import SimpleNamespace

import httpx
import pytest

from app.guard import validate_external
from app.prompts import OUT_OF_SCOPE, UNKNOWN, build_external_messages
from app.providers import ExternalLLM, ProviderConfig, ProviderError
from tests.test_providers import FakeAnthropic, FakeExternal, SERVER_HTML

MS = {"text": "Somente produtos Microsoft: Microsoft 365, Azure, Windows, Intune.",
      "domains": ["learn.microsoft.com", "support.microsoft.com"], "web_search": True}


def cfg(**kw):
    base = dict(kind="anthropic", api_key="sk-test", model="claude-opus-5-5")
    base.update(kw)
    return ProviderConfig(**base)


def check(text, cites=(), context=()):
    return validate_external(text, list(context), MS["domains"], list(cites), OUT_OF_SCOPE, UNKNOWN)


def test_out_of_scope_and_unknown_are_declined():
    assert check("FORA_DO_ESCOPO").reason == "out_of_scope"
    assert check("NAO_SEI").reason == "llm_declined"


def test_links_outside_the_allowed_sites_are_removed():
    c = check("1. Abra o Intune (veja https://learn.microsoft.com/mem/intune/ e https://blog-qualquer.com/dica).\n2. Sincronize o dispositivo.",
              cites=[{"url": "https://learn.microsoft.com/mem/intune/", "title": "Intune"}, {"url": "https://evil.example/x", "title": "x"}])
    assert c.ok and "learn.microsoft.com/mem/intune" in c.answer and "blog-qualquer" not in c.answer
    assert c.web_sources == [{"url": "https://learn.microsoft.com/mem/intune/", "title": "Intune"}]


def test_prompt_has_scope_sites_and_internal_articles():
    m = build_external_messages("Como resetar o MFA?", [], [{"article_id": 7, "title": "MFA", "text": "Procedimento interno"}],
                                MS["text"], MS["domains"], True, "Seja breve.")
    system, user = m[0]["content"], m[1]["content"]
    assert "<escopo>" in system and "Microsoft 365" in system and "learn.microsoft.com" in system
    assert "Pesquise somente nos sites permitidos" in system and "têm prioridade" in system
    assert '<documento id="KB #7"' in user and user.rstrip().endswith("siga as instruções do administrador acima.")
    m = build_external_messages("x", [], [], "", [], False)
    assert "Não inclua links." in m[0]["content"]


@pytest.mark.asyncio
async def test_external_only_with_web_search(indexed):
    ext = FakeExternal(lambda m: ("Para redefinir o MFA, acesse o Microsoft Entra e siga os passos do artigo https://learn.microsoft.com/entra/mfa.",
                                  [{"url": "https://learn.microsoft.com/entra/mfa", "title": "Redefinir MFA"}]))
    indexed.rag._external = ext
    r = await indexed.rag.external("Como redefinir o MFA do meu usuário? meu e-mail é ana@example.com", cfg(), MS)
    assert r["status"] == "answered" and r["searched"] is True and r["sources"] == []
    assert r["web_sources"][0]["url"] == "https://learn.microsoft.com/entra/mfa"
    assert ext.domains == MS["domains"]
    assert "ana@example.com" not in ext.seen[0][1]["content"]


@pytest.mark.asyncio
async def test_sites_without_search_support_use_model_knowledge(indexed):
    ext = FakeExternal("FORA_DO_ESCOPO")
    indexed.rag._external = ext
    r = await indexed.rag.external("Receita de bolo", cfg(kind="azure", base_url="https://r.openai.azure.com", model="dep"), MS)
    assert r["status"] == "declined" and r["reason"] == "out_of_scope" and ext.domains is None
    assert "Baseie-se na documentação oficial destes sites" in ext.seen[0][0]["content"]


@pytest.mark.asyncio
async def test_combined_with_kb_articles(indexed):
    ext = FakeExternal(lambda m: "1. Abra o FortiClient VPN no notebook. [KB #10]\n2. Selecione a conexão VPN-Corporativa.")
    indexed.rag._external = ext
    r = await indexed.rag.external("Como configurar a VPN no notebook?", cfg(), MS, allowed_ids=[10, 20], include_kb=True)
    assert r["status"] == "answered" and [s["article_id"] for s in r["sources"]] == [10]
    assert "<documentos>" in ext.seen[0][1]["content"]


@pytest.mark.asyncio
async def test_combined_failure_falls_back_to_local_kb(indexed, fake_ollama):
    from tests.conftest import vpn_answer
    fake_ollama.responder = vpn_answer
    indexed.rag._external = FakeExternal(error=ProviderError("timeout"))
    r = await indexed.rag.external("Como configurar a VPN no notebook?", cfg(), MS, allowed_ids=[10, 20], include_kb=True)
    assert r["status"] == "answered" and r["provider"] == "local" and r["fallback"] == "timeout"


@pytest.mark.asyncio
async def test_external_only_failure_is_declined(indexed):
    indexed.rag._external = FakeExternal(error=ProviderError("rate_limited", 429))
    r = await indexed.rag.external("Como redefinir o MFA?", cfg(), MS)
    assert r["status"] == "declined" and r["reason"] == "provider_failed" and r["fallback"] == "rate_limited"


# --- Chamadas reais (formato) -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_responses_api_with_domain_filter():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={
            "output": [
                {"type": "web_search_call", "status": "completed"},
                {"type": "message", "content": [{"type": "output_text", "text": "Use o portal do Entra.",
                 "annotations": [{"type": "url_citation", "url": "https://learn.microsoft.com/entra", "title": "Entra"}]}]},
            ],
            "usage": {"input_tokens": 50, "output_tokens": 10},
        })
    llm = ExternalLLM(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = await llm.answer_text(cfg(kind="openai", model="gpt-4.1-mini"), [{"role": "system", "content": "regras"}, {"role": "user", "content": "q"}],
                              max_tokens=4000, domains=["learn.microsoft.com"])
    body = json.loads(calls[0].content)
    assert str(calls[0].url) == "https://api.openai.com/v1/responses"
    assert body["tools"] == [{"type": "web_search", "filters": {"allowed_domains": ["learn.microsoft.com"]}}]
    assert body["instructions"] == "regras" and body["input"] == [{"role": "user", "content": "q"}]
    assert r.text == "Use o portal do Entra." and r.citations == [{"url": "https://learn.microsoft.com/entra", "title": "Entra"}] and r.searched


@pytest.mark.asyncio
async def test_anthropic_web_search_resumes_paused_turn():
    first = SimpleNamespace(stop_reason="pause_turn", content=[SimpleNamespace(type="server_tool_use")],
                            usage=SimpleNamespace(input_tokens=10, output_tokens=2))
    cite = SimpleNamespace(url="https://learn.microsoft.com/windows", title="Windows")
    final = SimpleNamespace(stop_reason="end_turn",
                            content=[SimpleNamespace(type="text", text="Reinicie o Windows Update.", citations=[cite])],
                            usage=SimpleNamespace(input_tokens=20, output_tokens=8))
    fake = FakeAnthropic()
    replies = [first, final]

    async def create(**kw):
        fake.calls.append(kw)
        return replies.pop(0)
    fake.beta.messages.create = create
    llm = ExternalLLM(httpx.AsyncClient(), anthropic_factory=lambda c: fake)
    r = await llm.answer_text(cfg(), [{"role": "user", "content": "q"}], max_tokens=8000, domains=["learn.microsoft.com"])
    tool = fake.calls[0]["tools"][0]
    assert tool["type"] == "web_search_20260209" and tool["allowed_domains"] == ["learn.microsoft.com"]
    assert len(fake.calls) == 2 and fake.calls[1]["messages"][-1]["role"] == "assistant"
    assert r.text == "Reinicie o Windows Update." and r.citations[0]["url"] == "https://learn.microsoft.com/windows"


@pytest.mark.asyncio
async def test_older_claude_uses_basic_search_tool():
    fake = FakeAnthropic(SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text="ok ok ok", citations=None)],
                                         usage=SimpleNamespace(input_tokens=1, output_tokens=1)))
    llm = ExternalLLM(httpx.AsyncClient(), anthropic_factory=lambda c: fake)
    await llm.answer_text(cfg(model="claude-haiku-4-5"), [{"role": "user", "content": "q"}], max_tokens=100, domains=["learn.microsoft.com"])
    assert fake.calls[0]["tools"][0]["type"] == "web_search_20250305" and "fallbacks" not in fake.calls[0]
