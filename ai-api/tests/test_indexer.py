"""Sincronização da KB: reindexação, exclusão, alteração e falhas do GLPI."""

import pytest

from app.errors import GlpiUnavailable, SyncAlreadyRunning
from tests.conftest import vpn_answer

pytestmark = pytest.mark.asyncio


async def test_full_reindex_switches_collection(container):
    """11. Reindexação completa troca a coleção sem perder o índice."""
    r1 = await container.indexer.full()
    first = await container.store.active_collection()
    r2 = await container.indexer.full()
    second = await container.store.active_collection()
    assert r1["articles_indexed"] == r2["articles_indexed"] == 2
    assert first != second
    collections = [c.name for c in (await container.store._client.get_collections()).collections]
    assert first not in collections  # coleção antiga removida
    assert (await container.store.counts())["articles"] == 2


async def test_sync_removes_deleted_article(indexed, fake_glpi, fake_ollama):
    """12. Exclusão de artigo no GLPI remove do índice."""
    del fake_glpi.articles[10]
    r = await indexed.indexer.sync()
    assert r["deleted"] == 1
    res = await indexed.rag.search("Como configuro a VPN no notebook?")
    assert 10 not in [c["article_id"] for c in res["candidates"]]
    fake_ollama.responder = vpn_answer
    chat = await indexed.rag.chat("Como configuro a VPN no notebook?", [10, 20])
    assert chat.status == "no_evidence"


async def test_sync_updates_changed_article(indexed, fake_glpi):
    """13. Alteração de artigo: novo conteúdo indexado, antigo descartado."""
    r = await indexed.indexer.sync()
    assert r["articles_updated"] == 0  # nada mudou
    fake_glpi.add(20, "Impressora do financeiro não imprime",
                  "<p>Abra um chamado para a equipe de infraestrutura trocar o toner da Lexmark.</p>")
    r = await indexed.indexer.sync()
    assert r["articles_updated"] == 1
    hits = await indexed.rag.retrieve("toner Lexmark", 5)
    texts = " ".join(h.text for h in hits if h.article_id == 20)
    assert "Lexmark" in texts and "HP do financeiro" not in texts


async def test_sync_adds_new_article(indexed, fake_glpi):
    fake_glpi.add(40, "Acesso ao ERP", "<p>Para acessar o ERP use o atalho ERP Produção na área de trabalho.</p>")
    r = await indexed.indexer.sync()
    assert r["articles_updated"] == 1
    assert (await indexed.store.counts())["articles"] == 3


async def test_index_specific_articles(indexed, fake_glpi):
    fake_glpi.add(20, "Impressora", "<p>Nova orientação sobre a impressora Epson.</p>")
    del fake_glpi.articles[10]
    r = await indexed.indexer.index_articles([10, 20])
    assert r == {"mode": "articles", "articles_updated": 1, "deleted": 1}


async def test_glpi_failure_keeps_existing_index(indexed, fake_glpi, fake_ollama):
    """8. Falha da API do GLPI na sincronização não apaga o índice atual."""
    fake_glpi.fail = True
    with pytest.raises(GlpiUnavailable):
        await indexed.indexer.sync()
    with pytest.raises(GlpiUnavailable):
        await indexed.indexer.full()
    await indexed.indexer.record_error("sync", GlpiUnavailable("x"))
    state = await indexed.store.get_state()
    assert state["last_error"]["code"] == "glpi_unavailable"
    fake_ollama.responder = vpn_answer
    r = await indexed.rag.chat("Como configuro a VPN no notebook?", [10, 20])
    assert r.status == "answered"
    # nenhuma coleção órfã ficou para trás
    collections = [c.name for c in (await indexed.store._client.get_collections()).collections]
    assert len([c for c in collections if c.startswith("kb_test_1")]) == 1


async def test_concurrent_sync_is_rejected(indexed):
    async with indexed.indexer._exclusive("full"):
        with pytest.raises(SyncAlreadyRunning):
            await indexed.indexer.sync()


async def test_embedding_model_change_forces_full(indexed):
    indexed.settings.embed_model = "outro-modelo"
    try:
        r = await indexed.indexer.sync()
        assert r["mode"] == "full"
    finally:
        indexed.settings.embed_model = "qwen3-embedding:0.6b"


async def test_expired_article_is_not_used(container, fake_glpi):
    fake_glpi.add(50, "Procedimento antigo do proxy", "<p>Configure o proxy antigo 10.0.0.1 no navegador.</p>",
                  date_end="2001-01-01 00:00:00")
    await container.indexer.full()
    res = await container.rag.search("proxy antigo navegador")
    assert 50 not in [c["article_id"] for c in res["candidates"]]
