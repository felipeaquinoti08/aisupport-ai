# aisupport-ai

Pilha local de IA do **Agente de Suporte N1** para GLPI 11: Ollama (LLM e embeddings), Qdrant (banco vetorial) e ai-api (FastAPI, RAG). Funciona em ARM64 (Oracle Ampere) e x86_64, sem GPU e sem nenhum serviço externo.

O plugin do GLPI fica em outro repositório: [aisupport-](https://github.com/felipeaquinoti08/aisupport-).

> Documentação completa na Fase 15. Este README cobre o deploy mínimo.

## Deploy rápido

```bash
git clone https://github.com/felipeaquinoti08/aisupport-ai.git && cd aisupport-ai
cp .env.example .env && chmod 600 .env     # preencha AI_API_KEY, QDRANT_API_KEY e GLPI_*
docker compose up -d
docker compose --profile setup run --rm model-puller   # baixa os modelos (só esta etapa usa Internet)
docker compose exec ollama ollama list
curl -s http://127.0.0.1:8080/api/live
```

## Rede e isolamento

| Serviço | Rede | Internet | Porta publicada |
|---|---|---|---|
| ollama | `glpi_ai_net` (interna) | não | não |
| qdrant | `glpi_ai_net` (interna) | não | não |
| ai-api | `glpi_ai_net` + `glpi_ai_edge` | sim (para alcançar o GLPI) | `AI_API_BIND:AI_API_PORT` |
| model-puller | `glpi_ai_edge` | sim, só durante o download | não |

Publique a ai-api apenas no IP alcançável pelo servidor do GLPI e libere no firewall somente esse servidor. O Ollama e o Qdrant nunca ficam expostos.

## Memória (limite de 4 GB para o conjunto)

| Serviço | Limite | Medido (2 modelos carregados) |
|---|---|---|
| ollama | 3456m | ~3,0 GB |
| qdrant | 320m | ~50 MB (KB pequena) |
| ai-api | 320m | ~80 MB |

## API (contrato 1.0)

Todas as rotas, exceto `/api/live`, exigem `Authorization: Bearer <AI_API_KEY>`.

| Método | Rota | Uso |
|---|---|---|
| GET | `/api/live` | Liveness do container (sem autenticação) |
| GET | `/api/health` | Estado de Ollama, modelos, Qdrant e acesso ao GLPI |
| GET | `/api/status` | Versão do contrato, modelos, parâmetros do RAG e estado do índice |
| POST | `/api/search` | Candidatos (id, título, score) para o plugin checar permissões |
| POST | `/api/chat` | Resposta restrita aos `allowed_article_ids` autorizados pelo plugin |
| POST | `/api/summarize` | Título e resumo para abertura de chamado |
| POST | `/api/test-llm` | Teste rápido do modelo |
| POST | `/api/index` | Reindexa artigos específicos (criação, alteração, exclusão) |
| POST | `/api/sync` | Sincronização incremental em segundo plano |
| POST | `/api/reindex` | Reindexação completa em segundo plano (troca atômica da coleção) |

## Comandos úteis

```bash
docker compose exec ai-api python -m app.indexer.full    # reindexação completa
docker compose exec ai-api python -m app.indexer.sync    # sincronização incremental
docker compose exec -T ai-api python -m app.calibrate < perguntas.jsonl   # calibrar RAG_MIN_SCORE
```

## Como o agente decide responder

1. Busca híbrida (vetorial + palavras-chave) só entre artigos que o usuário pode ver.
2. Sem evidência (score abaixo de `RAG_MIN_SCORE` ou baixa cobertura dos termos) o LLM nem é chamado.
3. O LLM gera JSON sob schema (`resposta`, `encontrado`).
4. A resposta é descartada se: `encontrado=false`, citar artigo fora do contexto, trazer URL que não está no documento ou tiver baixa sobreposição com o texto dos artigos.
5. As fontes exibidas são os artigos que de fato sustentam a resposta.

## Testes

```bash
cd ai-api
docker run --rm -v "$PWD":/src -w /src python:3.13-slim sh -c \
  "pip install -q -r requirements-dev.txt && python -m pytest -q"
```
