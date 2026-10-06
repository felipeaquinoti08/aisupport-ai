"""Provedor externo (OpenAI, Azure, compatível, Anthropic): formato das chamadas,
mascaramento de dados pessoais e queda para o modelo local."""

import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.errors import LlmUnavailable
from app.prompts import ANSWER_SCHEMA
from app.providers import ExternalLLM, PiiMasker, ProviderConfig, ProviderError, unmask_json_text
from app.rag import ChatOptions
from tests.conftest import llm_json, vpn_answer

MSGS = [{"role": "system", "content": "Responda em JSON."}, {"role": "user", "content": "Olá"}]


def openai_reply(content, status=200, finish="stop"):
    if status != 200:
        return httpx.Response(status, json={"error": {"message": content}})
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 7},
    })


def recording_client(*replies):
    calls = []
    queue = list(replies)

    def handler(request: httpx.Request):
        calls.append(request)
        return queue.pop(0)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


def cfg(**kw):
    base = dict(kind="openai", api_key="sk-test-123", model="gpt-test")
    base.update(kw)
    return ProviderConfig(**base)


# --- OpenAI e compatíveis ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_openai_uses_strict_json_schema():
    http, calls = recording_client(openai_reply('{"resposta": "x", "encontrado": true}'))
    r = await ExternalLLM(http).chat(cfg(), MSGS, schema=ANSWER_SCHEMA, max_tokens=4000, temperature=0.1)
    assert r.text.startswith('{"resposta"') and r.model == "openai:gpt-test" and r.prompt_tokens == 12
    req = calls[0]
    body = json.loads(req.content)
    assert str(req.url) == "https://api.openai.com/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test-123"
    assert body["model"] == "gpt-test" and body["max_completion_tokens"] == 4000 and body["temperature"] == 0.1
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_unsupported_options_are_retried_without_them():
    http, calls = recording_client(
        openai_reply("Unsupported parameter: 'temperature' is not supported with this model.", 400),
        openai_reply("response_format json_schema is not supported", 400),
        openai_reply('{"ok": true}'),
    )
    r = await ExternalLLM(http).chat(cfg(kind="compatible", base_url="https://llm.example.com/v1"), MSGS,
                                     schema={"type": "object", "properties": {"ok": {"type": "boolean"}}}, max_tokens=100, temperature=0.2)
    assert r.text == '{"ok": true}'
    bodies = [json.loads(c.content) for c in calls]
    assert str(calls[0].url) == "https://llm.example.com/v1/chat/completions"
    assert "temperature" not in bodies[1]
    assert bodies[2]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("status, code", [(401, "auth_failed"), (404, "model_not_found"), (429, "rate_limited"), (503, "provider_error")])
@pytest.mark.asyncio
async def test_http_errors_become_stable_codes(status, code):
    http, _ = recording_client(openai_reply("erro", status))
    with pytest.raises(ProviderError) as e:
        await ExternalLLM(http).chat(cfg(), MSGS, schema=None, max_tokens=10)
    assert e.value.code == code


@pytest.mark.asyncio
async def test_content_filter_is_a_refusal():
    http, _ = recording_client(openai_reply("", finish="content_filter"))
    with pytest.raises(ProviderError) as e:
        await ExternalLLM(http).chat(cfg(), MSGS, schema=None, max_tokens=10)
    assert e.value.code == "refused"


@pytest.mark.parametrize("base_url, version, url, model_in_body", [
    ("https://res.openai.azure.com", "2024-10-21",
     "https://res.openai.azure.com/openai/deployments/meu-gpt/chat/completions?api-version=2024-10-21", False),
    ("https://res.openai.azure.com/", "", "https://res.openai.azure.com/openai/v1/chat/completions", True),
    ("https://res.services.ai.azure.com/models", "",
     "https://res.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview", True),
])
@pytest.mark.asyncio
async def test_azure_endpoints(base_url, version, url, model_in_body):
    http, calls = recording_client(openai_reply('{"ok": true}'))
    await ExternalLLM(http).chat(cfg(kind="azure", base_url=base_url, api_version=version, model="meu-gpt"), MSGS, schema=None, max_tokens=10)
    assert str(calls[0].url) == url
    assert calls[0].headers["api-key"] == "sk-test-123" and "authorization" not in calls[0].headers
    assert ("model" in json.loads(calls[0].content)) is model_in_body


# --- Anthropic ----------------------------------------------------------------------------

class FakeAnthropic:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.closed = False
        outer = self

        class _Messages:
            async def create(self, **kw):
                outer.calls.append(kw)
                if error:
                    raise error
                return response

        self.messages = _Messages()
        self.beta = SimpleNamespace(messages=_Messages())

    async def close(self):
        self.closed = True


def anthropic_response(text, stop="end_turn"):
    return SimpleNamespace(
        stop_reason=stop,
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        usage=SimpleNamespace(input_tokens=30, output_tokens=9),
    )


@pytest.mark.asyncio
async def test_anthropic_structured_output_with_server_fallback():
    fake = FakeAnthropic(anthropic_response('{"resposta": "x", "encontrado": true}'))
    llm = ExternalLLM(httpx.AsyncClient(), anthropic_factory=lambda c: fake)
    r = await llm.chat(cfg(kind="anthropic", model="claude-opus-5-5", effort="low"), MSGS, schema=ANSWER_SCHEMA, max_tokens=4000, temperature=0.1)
    call = fake.calls[0]
    assert r.text.startswith('{"resposta"') and r.model == "anthropic:claude-opus-5-5" and fake.closed
    assert call["system"] == "Responda em JSON." and call["messages"] == [{"role": "user", "content": "Olá"}]
    assert call["output_config"]["format"]["type"] == "json_schema" and call["output_config"]["effort"] == "low"
    assert call["betas"] == ["server-side-fallback-2026-07-01"] and call["fallbacks"] == "default"
    assert "temperature" not in call  # rejeitado pelos modelos atuais


@pytest.mark.asyncio
async def test_anthropic_older_model_has_no_fallbacks_and_refusal_is_reported():
    fake = FakeAnthropic(anthropic_response("", stop="refusal"))
    llm = ExternalLLM(httpx.AsyncClient(), anthropic_factory=lambda c: fake)
    with pytest.raises(ProviderError) as e:
        await llm.chat(cfg(kind="anthropic", model="claude-haiku-4-5"), MSGS, schema=None, max_tokens=100)
    assert e.value.code == "refused" and "fallbacks" not in fake.calls[0]


@pytest.mark.asyncio
async def test_anthropic_auth_error():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.AuthenticationError("invalid x-api-key", response=httpx.Response(401, request=req), body=None)
    llm = ExternalLLM(httpx.AsyncClient(), anthropic_factory=lambda c: FakeAnthropic(error=err))
    with pytest.raises(ProviderError) as e:
        await llm.chat(cfg(kind="anthropic", model="claude-opus-5-5"), MSGS, schema=None, max_tokens=100)
    assert e.value.code == "auth_failed"


# --- Dados pessoais ---------------------------------------------------------------------------

def test_masker_round_trip():
    m = PiiMasker()
    text = "Sou a ana.souza@example.com, CPF 123.456.789-09, fone (11) 98765-4321, servidor 203.0.113.10; de novo ana.souza@example.com"
    masked = m.mask(text)
    for value in ("ana.souza@example.com", "123.456.789-09", "98765-4321", "203.0.113.10"):
        assert value not in masked
    assert masked.count("[EMAIL_1]") == 2 and "[CPF_1]" in masked and "[IP_1]" in masked and "[TELEFONE_1]" in masked
    assert m.unmask(masked) == text
    assert json.loads(unmask_json_text(json.dumps({"resposta": "Use [IP_1]"}), m))["resposta"] == "Use 203.0.113.10"


# --- Integração com o RAG -----------------------------------------------------------------------

class FakeExternal:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.seen = reply, error, []

    async def chat(self, provider, messages, *, schema, max_tokens, temperature=None):
        from app.ollama import LlmResult
        self.seen.append(messages)
        if self.error:
            raise self.error
        text = self.reply(messages) if callable(self.reply) else self.reply
        return LlmResult(text=text, model=provider.label, prompt_tokens=1, output_tokens=1, duration_ms=5)

    async def answer_text(self, provider, messages, *, max_tokens, domains=None, temperature=None):
        from app.providers import TextResult
        self.seen.append(messages)
        self.domains = domains
        if self.error:
            raise self.error
        reply = self.reply(messages) if callable(self.reply) else self.reply
        text, cites = reply if isinstance(reply, tuple) else (reply, [])
        return TextResult(text=text, model=provider.label, citations=cites, prompt_tokens=1, output_tokens=1,
                          duration_ms=5, searched=bool(domains))

    async def aclose(self):
        pass


SERVER_HTML = "<p>Para acessar o ERP use o servidor 203.0.113.50 e, em caso de erro, escreva para suporte.erp@example.com.</p>"


@pytest.mark.asyncio
async def test_external_answer_with_masked_data(container, fake_glpi, fake_ollama):
    fake_glpi.add(30, "Acesso ao ERP", SERVER_HTML)
    await container.indexer.full()
    ext = FakeExternal(lambda m: llm_json("Para acessar o ERP use o servidor [IP_1] e, em caso de erro, escreva para [EMAIL_1]."))
    container.rag._external = ext
    r = await container.rag.chat("Como acessar o ERP?", [10, 20, 30], provider=cfg())

    sent = ext.seen[0][1]["content"]
    assert "203.0.113.50" not in sent and "suporte.erp@example.com" not in sent and "[IP_1]" in sent
    assert r.status == "answered" and r.provider == "openai:gpt-test" and r.fallback == ""
    assert "203.0.113.50" in r.answer and "suporte.erp@example.com" in r.answer
    assert fake_ollama.calls == [], "the local model only did the search"


@pytest.mark.asyncio
async def test_provider_failure_falls_back_to_local_model(indexed, fake_ollama):
    indexed.rag._external = FakeExternal(error=ProviderError("rate_limited", 429))
    fake_ollama.responder = vpn_answer
    r = await indexed.rag.chat("Como configurar a VPN no notebook?", [10, 20], provider=cfg())
    assert r.status == "answered" and r.provider == "local" and r.fallback == "rate_limited"
    assert len(fake_ollama.calls) == 1


@pytest.mark.asyncio
async def test_without_local_fallback_the_failure_is_reported(indexed):
    indexed.rag._external = FakeExternal(error=ProviderError("auth_failed", 401))
    with pytest.raises(LlmUnavailable):
        await indexed.rag.chat("Como configurar a VPN no notebook?", [10, 20], provider=cfg(fallback_local=False))


@pytest.mark.asyncio
async def test_external_answer_is_still_validated_against_the_kb(indexed):
    # Resposta fora dos documentos continua sendo descartada
    indexed.rag._external = FakeExternal(llm_json("Reinstale o Windows e formate o disco para resolver a VPN."))
    r = await indexed.rag.chat("Como configurar a VPN no notebook?", [10, 20], provider=cfg())
    assert r.status == "no_evidence" and r.reason == "low_grounding"


@pytest.mark.asyncio
async def test_mask_can_be_disabled(container, fake_glpi):
    fake_glpi.add(30, "Acesso ao ERP", SERVER_HTML)
    await container.indexer.full()
    ext = FakeExternal(llm_json("Use o servidor 203.0.113.50 para acessar o ERP."))
    container.rag._external = ext
    await container.rag.chat("Como acessar o ERP?", [30], provider=cfg(mask_pii=False))
    assert "203.0.113.50" in ext.seen[0][1]["content"]


@pytest.mark.asyncio
async def test_summary_and_general_use_the_provider(indexed, fake_ollama):
    ext = FakeExternal(lambda m: '{"titulo": "Erro no ERP", "resumo": "Usuário sem acesso."}'
                       if "titulo" in m[0]["content"] else '{"resposta": "1. Reinicie o computador.", "respondeu": true}')
    indexed.rag._external = ext
    s = await indexed.rag.summarize("Não consigo acessar o ERP", [], provider=cfg())
    g = await indexed.rag.general("Computador lento", provider=cfg())
    assert s["provider"] == "openai:gpt-test" and s["generated"] is True
    assert g["status"] == "answered" and g["provider"] == "openai:gpt-test"
    assert fake_ollama.calls == []


@pytest.mark.asyncio
async def test_provider_test_reports_errors(indexed):
    indexed.rag._external = FakeExternal('{"ok": true}')
    assert (await indexed.rag.test_provider(cfg()))["ok"] is True
    indexed.rag._external = FakeExternal(error=ProviderError("auth_failed", 401))
    assert (await indexed.rag.test_provider(cfg())) == {"ok": False, "error": "auth_failed", "status": 401, "detail": ""}


@pytest.mark.asyncio
async def test_options_still_work_without_provider(indexed, fake_ollama):
    fake_ollama.responder = vpn_answer
    r = await indexed.rag.chat("Como configurar a VPN no notebook?", [10, 20], opts=ChatOptions())
    assert r.status == "answered" and r.provider == "local"
