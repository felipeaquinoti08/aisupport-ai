# aisupport-ai

Pilha local de IA do **Agente de Suporte N1** para GLPI 11. São três serviços: **Ollama** (LLM e embeddings), **Qdrant** (banco vetorial) e **ai-api** (FastAPI com RAG e indexação da Base de Conhecimento). Roda em **ARM64** (Oracle Ampere) ou x86_64, só com CPU, em **4 GB de RAM** no total, e **sem nenhum serviço externo** durante o funcionamento.

O plugin do GLPI fica em [aisupport-](https://github.com/felipeaquinoti08/aisupport-).

---

## Sumário

- [Arquitetura](#arquitetura)
- [Requisitos](#requisitos)
- [Modelo e embeddings](#modelo-e-embeddings)
- [Instalação](#instalação)
- [Deploy no Komodo](#deploy-no-komodo)
- [Configuração (.env)](#configuração-env)
- [Conta da API do GLPI](#conta-da-api-do-glpi)
- [Indexação da Base de Conhecimento](#indexação-da-base-de-conhecimento)
- [RAG e calibração do threshold](#rag-e-calibração-do-threshold)
- [API (contrato 1.0)](#api-contrato-10)
- [Rede e segurança](#rede-e-segurança)
- [Testes](#testes)
- [Troubleshooting](#troubleshooting)
- [Backup](#backup)
- [Atualização](#atualização)
- [Remoção](#remoção)

---

## Arquitetura

```
servidor do GLPI ──HTTPS + Bearer AI_API_KEY──► ai-api :8080 ──┬── ollama :11434  (glpi_ai_net, interna, sem Internet)
                                                  │            └── qdrant :6333   (glpi_ai_net, interna, sem Internet)
                                                  └──OAuth2──► API REST v2 do GLPI (leitura da KB)
```

| Serviço | Rede | Internet | Porta publicada | Memória |
|---|---|---|---|---|
| ollama | `glpi_ai_net` (internal) | não | não | 3456m |
| qdrant | `glpi_ai_net` (internal) | não | não | 320m |
| ai-api | `glpi_ai_net` + `glpi_ai_edge` | só para alcançar o GLPI | `AI_API_BIND:AI_API_PORT` | 320m |
| model-puller | `glpi_ai_edge` | sim, só no download dos modelos | não | — |

- A rede `glpi_ai_net` é `internal: true`: o Ollama e o Qdrant **não conseguem acessar a Internet** e não ficam expostos. Além disso, `OLLAMA_NO_CLOUD=1` e a telemetria do Qdrant ficam desligadas.
- Os modelos são baixados por um serviço separado e temporário (`model-puller`), o único com saída. O Ollama de produção nunca acessa a Internet.
- A ai-api roda **sem root**, com sistema de arquivos **somente leitura** e sem capabilities.

---

## Requisitos

- Linux ARM64 ou x86_64 com **Docker** e **Docker Compose v2**.
- **4 GB de RAM** livres para a pilha (soma dos limites) e cerca de **5 GB de disco** (imagens + 2,5 GB de modelos).
- **2 a 4 vCPUs.** Com mais núcleos, a resposta fica mais rápida.
- Rede: o servidor do GLPI precisa alcançar a ai-api, e a ai-api precisa alcançar a URL do GLPI.

Instalação do Docker (Ubuntu/Oracle Linux, ARM64 ou x86):

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # e faça login de novo
docker compose version
```

---

## Modelo e embeddings

| | Escolha | Tamanho | Quantização | RAM medida |
|---|---|---|---|---|
| **LLM** | `qwen2.5:3b-instruct-q4_K_M` | 1,9 GB | Q4_K_M | ~2,1 GB carregado (contexto 4096, KV cache q8_0) |
| **Embeddings** | `qwen3-embedding:0.6b` | 639 MB | Q8 | ~1,0 GB carregado (contexto 512) |
| **Total do Ollama** | | | | **~2,9–3,0 GB** com os dois carregados (limite 3456m) |

**Por que esses:**
- O Qwen2.5-3B tem o melhor português e a melhor obediência a instruções da faixa de 3B, e cabe no orçamento de 4 GB **do conjunto**.
- O Qwen3-Embedding é multilíngue e forte em PT-BR. O `nomic-embed-text` é fraco em português e o `bge-m3` é pesado demais.
- A geração usa **saída JSON sob schema**, porque modelos pequenos erram menos com saída estruturada. Nos testes, ela eliminou respostas deformadas e recusas indevidas.

**Desempenho esperado** (CPU, sem GPU):

| Etapa | x86 compartilhado (medido) | Ampere A1 4 OCPU (estimado) |
|---|---|---|
| Recusa sem evidência | < 10 ms | < 10 ms |
| Busca (embedding + Qdrant) | 10–400 ms | 100–400 ms |
| Resposta completa (até 3 trechos) | 6–16 s; 25–30 s com o modelo frio | 8–20 s |
| Geração | ~8 tokens/s | 10–15 tokens/s |

**Alternativas** (troque `LLM_MODEL` e rode o `model-puller` de novo):

| Modelo | Quando usar |
|---|---|
| `llama3.2:3b-instruct-q4_K_M` | Alternativa de 3B. Português um pouco inferior |
| `qwen2.5:1.5b-instruct-q4_K_M` | Servidor com menos RAM ou CPU. Mais rápido, menos preciso |
| `qwen3:4b-instruct` / `gemma3:4b` | Só se houver **mais de 4 GB** para o conjunto (~2,6–3,3 GB só o LLM) |

Para comparar modelos na sua KB, use `EXTRA_MODELS` no `.env`, baixe-os e mude `LLM_MODEL`, ou informe o modelo na configuração do plugin.

---

## Instalação

```bash
# 1. Código e diretório
git clone https://github.com/felipeaquinoti08/aisupport-ai.git /opt/aisupport-ai
cd /opt/aisupport-ai

# 2. Configuração
cp .env.example .env && chmod 600 .env
openssl rand -hex 32     # use em AI_API_KEY (a mesma vai no plugin)
openssl rand -hex 32     # use em QDRANT_API_KEY
nano .env                # AI_API_KEY, QDRANT_API_KEY, AI_API_BIND, GLPI_* (ver abaixo)

# 3. Subir
docker compose up -d
docker compose ps        # ollama, qdrant e ai-api "healthy"

# 4. Baixar os modelos (única etapa que usa Internet)
docker compose --profile setup run --rm model-puller

# 5. Conferir os modelos e o Ollama
docker compose exec ollama ollama list
docker compose exec ollama ollama run qwen2.5:3b-instruct-q4_K_M "Responda apenas: OK"

# 6. Testar a ai-api
curl -s http://127.0.0.1:8080/api/live
curl -s -H "Authorization: Bearer $(grep ^AI_API_KEY= .env | cut -d= -f2)" http://127.0.0.1:8080/api/health

# 7. Indexar a KB (também acontece sozinho ~20 s após subir, e pelo botão no GLPI)
docker compose exec ai-api python -m app.indexer.full
```

O Ollama **não é publicado**. Para consultar as tags dele, use `docker compose exec ollama ollama list`. A API do Ollama (`/api/tags`) só é acessível de dentro da rede interna, por exemplo com `docker compose exec ai-api python -c "import urllib.request;print(urllib.request.urlopen('http://ollama:11434/api/tags').read()[:200])"`.

Para o Komodo (ou Portainer), veja [Deploy no Komodo](#deploy-no-komodo). Lá o download dos modelos também acontece pelo Docker, no próprio deploy.

Depois disso, configure o plugin no GLPI (aba **Servidor de IA**) com a URL e a `AI_API_KEY`, e clique em **[Testar conexão]**.

## Deploy no Komodo

A pilha inteira roda em Docker, **inclusive o download do LLM**: com `COMPOSE_PROFILES=setup`, o serviço `model-puller` sobe junto em cada deploy, baixa os modelos que faltam (os que já existem são pulados) e encerra. Não é preciso abrir terminal no servidor.

### 1. Servidor

O servidor de IA (Oracle ARM64) precisa ter o **Komodo Periphery** conectado ao seu Komodo Core e o Docker instalado. Ele aparece em **Servers** como *Ok*.

### 2. Variáveis secretas (recomendado)

Em **Settings > Variables**, crie as variáveis abaixo marcadas como **secret**. Assim elas não aparecem na tela da stack nem nos logs.

| Variável | Valor |
|---|---|
| `AISUPPORT_AI_API_KEY` | `openssl rand -hex 32` (a mesma vai no plugin do GLPI) |
| `AISUPPORT_QDRANT_API_KEY` | `openssl rand -hex 32` |
| `AISUPPORT_GLPI_CLIENT_SECRET` | segredo do cliente OAuth |
| `AISUPPORT_GLPI_PASSWORD` | senha da conta de serviço |

### 3. Stack

**Stacks > New Stack** (ex.: `aisupport-ai`):

| Campo | Valor |
|---|---|
| Server | o servidor de IA (ARM64) |
| Source | Git Repo: `felipeaquinoti08/aisupport-ai`, branch `main` (repositório público; account só se ele virar privado) |
| Compose file | `docker-compose.yml` (padrão) |
| Run Build | **ligado**: constrói a imagem da ai-api no próprio servidor (ARM64) a cada deploy |
| Auto Pull / Webhook | opcional, para redeploy ao dar push no `main` |

Em **Environment**, cole o ambiente. O Komodo grava esse conteúdo como `.env` na pasta da stack e o Compose o usa, tanto na interpolação quanto no `env_file` da ai-api:

```env
COMPOSE_PROJECT_NAME=glpi-ai
COMPOSE_PROFILES=setup

# Exposição da ai-api: IP privado da VM (VCN) ou da VPN, alcançável pelo GLPI.
# Restrinja a porta na Security List/NSG da Oracle e no firewall da VM.
AI_API_BIND=10.0.0.10
AI_API_PORT=8080
AI_API_KEY=[[AISUPPORT_AI_API_KEY]]
AI_API_ALLOWED_IPS=203.0.113.20/32

QDRANT_API_KEY=[[AISUPPORT_QDRANT_API_KEY]]

GLPI_URL=https://glpi.example.com
GLPI_OAUTH_CLIENT_ID=<id do cliente OAuth>
GLPI_OAUTH_CLIENT_SECRET=[[AISUPPORT_GLPI_CLIENT_SECRET]]
GLPI_USERNAME=agente-n1
GLPI_PASSWORD=[[AISUPPORT_GLPI_PASSWORD]]

LLM_MODEL=qwen2.5:3b-instruct-q4_K_M
EMBED_MODEL=qwen3-embedding:0.6b
RAG_MIN_SCORE=0.55
SYNC_INTERVAL_MINUTES=15
```

Os demais valores do [`.env.example`](.env.example) têm padrões e podem ficar de fora, como os limites de memória que somam 4 GiB. `AI_API_ALLOWED_IPS` deve conter o IP **de saída** do servidor do GLPI, do jeito que ele chega na VM de IA.

### 4. Deploy e primeiro uso

1. **Deploy.** No primeiro deploy, o `model-puller` baixa cerca de 2,5 GB de modelos. Acompanhe em **Logs > model-puller** até aparecer `Modelos disponíveis`. Em seguida ele fica *exited (0)*, o que é esperado.
2. Confira em **Services** que `ollama`, `qdrant` e `ai-api` estão **healthy**.
3. Uns 20 s depois de subir, a ai-api faz a primeira indexação da KB sozinha. Veja em **Logs > ai-api** o evento `index_full_done` (ou `index_failed`, com o motivo).
4. No GLPI: **Administração > Assistente de IA > Configurações > Servidor de IA**, informe `http://10.0.0.10:8080` (ou o endereço da VPN) e a mesma `AI_API_KEY`. Salve e clique em **[Testar conexão]**.

### Operação pelo Komodo

| Tarefa | Como |
|---|---|
| Atualizar a pilha | **Deploy** (com Run Build, a ai-api é reconstruída). Para o Ollama/Qdrant, mude `OLLAMA_IMAGE` ou `QDRANT_IMAGE` no Environment |
| Trocar ou adicionar modelo | Altere `LLM_MODEL` (ou `EXTRA_MODELS`) no Environment e faça **Deploy**. O `model-puller` baixa só o que falta |
| Reindexar a KB | Botão **[Reindexar tudo]** no painel do GLPI, ou no terminal do servidor: `docker compose -p glpi-ai exec ai-api python -m app.indexer.full` |
| Calibrar o threshold | Terminal do servidor: `docker compose -p glpi-ai exec -T ai-api python -m app.calibrate < perguntas.jsonl` |
| Ver consumo | **Stats** da stack ou do servidor. O Ollama fica em ~3 GB com os dois modelos carregados |
| Rodar 100% offline | Depois do primeiro download, deixe `COMPOSE_PROFILES` vazio. O `model-puller` não sobe mais e nada sai para a Internet |

> **Dica:** o servidor de IA não precisa de Internet depois do primeiro deploy. Se o Komodo ou o Periphery ainda precisarem acessar o GitHub para o *pull* do repositório, isso não afeta o Ollama nem o Qdrant, que continuam na rede interna sem saída.

---

## Configuração (.env)

| Variável | Padrão | Descrição |
|---|---|---|
| `COMPOSE_PROFILES` | vazio | `setup` = baixa os modelos que faltam a cada `up` (Komodo) |
| `AI_API_BIND` / `AI_API_PORT` | 127.0.0.1 / 8080 | IP e porta onde a ai-api é publicada. Use o IP da VPN ou da rede privada alcançável pelo GLPI |
| `AI_API_KEY` | — | Chave compartilhada com o plugin (mínimo de 32 caracteres) |
| `AI_API_ALLOWED_IPS` | vazio | CIDRs autorizados (ex.: `10.0.0.5/32`). Vazio = qualquer origem com a chave |
| `AI_API_TLS_CERT` / `AI_API_TLS_KEY` | vazio | TLS direto na ai-api (arquivos em `./certs`) |
| `GLPI_URL` | — | URL do GLPI alcançável pela ai-api |
| `GLPI_OAUTH_CLIENT_ID` / `_SECRET` | — | Cliente OAuth (concessão Password, escopo `api`) |
| `GLPI_USERNAME` / `GLPI_PASSWORD` | — | Conta que lê a KB |
| `GLPI_VERIFY_SSL` / `GLPI_CA_BUNDLE` | true / vazio | TLS até o GLPI |
| `GLPI_TIMEZONE` | America/Sao_Paulo | Fuso das datas de validade dos artigos |
| `LLM_MODEL` / `EMBED_MODEL` | qwen2.5:3b / qwen3-embedding:0.6b | Modelos |
| `LLM_NUM_CTX` / `EMBED_NUM_CTX` | 4096 / 512 | Contexto (afeta a RAM) |
| `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` / `LLM_TIMEOUT` | 0.1 / 512 / 180 | Geração |
| `RAG_MIN_SCORE` | 0.55 | Score mínimo de relevância. **Calibre** |
| `RAG_MIN_TERM_COVERAGE` | 0.25 | Fração mínima dos termos da pergunta presentes nos trechos |
| `RAG_MIN_ANSWER_OVERLAP` | 0.45 | Fração mínima da resposta presente nos documentos (fundamentação) |
| `RAG_TOP_K` / `RAG_CANDIDATES` | 3 / 20 | Trechos enviados ao LLM / candidatos para checagem de permissão |
| `RAG_MAX_CONTEXT_CHARS` | 4500 | Teto de contexto (tempo de resposta) |
| `RAG_EXCLUDE_SUSPICIOUS` | true | Exclui artigos com prompt injection |
| `SYNC_INTERVAL_MINUTES` | 15 | Sincronização incremental automática (0 = desligada) |
| `*_MEM_LIMIT` / `*_CPUS` | 3456m / 320m / 320m | Limites por serviço (soma = 4 GiB) |

O plugin pode ajustar, por requisição, o threshold, top_k, temperatura, max_tokens e modelo, sempre dentro de faixas seguras validadas aqui.

---

## Conta da API do GLPI

A ai-api lê a KB **somente** pela API REST v2 (`/api.php/token` e `/api.php/v2/Knowledgebase/Article`).

- Habilite a API v2 no GLPI (*Configurar > Geral > API*).
- Crie um cliente OAuth com concessão **Password** e escopo **api**.
- A conta precisa do direito **"Administração da base de conhecimento"** para enxergar **todos** os artigos, inclusive os restritos a grupos e perfis. **Quem vê o quê é decidido no plugin**, com as regras do GLPI aplicadas ao usuário da conversa. O que não for visível para a conta simplesmente não é indexado.

---

## Indexação da Base de Conhecimento

| Como | Quando |
|---|---|
| Automática | ~20 s após subir e a cada `SYNC_INTERVAL_MINUTES` (incremental, por hash do conteúdo) |
| Em tempo real | O plugin envia `POST /api/index` quando um artigo é criado, alterado ou excluído |
| Manual | `python -m app.indexer.sync` (incremental) ou `python -m app.indexer.full` (completa), ou os botões do painel no GLPI |

- **Incremental:** reindexa os artigos novos ou alterados e remove os que deixaram de existir.
- **Completa:** cria uma coleção nova, indexa tudo e **troca o alias** no final. O chat continua respondendo durante o processo. Se falhar, a coleção anterior continua ativa.
- Trocar o `EMBED_MODEL` força uma reindexação completa na próxima sincronização.
- Artigos com texto de instrução para a IA (*prompt injection*) são marcados como **suspeitos** e ficam fora das respostas. Eles aparecem no painel do GLPI.

---

## RAG e calibração do threshold

Fluxo de `/api/chat`:

1. Busca híbrida (vetorial + BM25 local), restrita aos `allowed_article_ids` enviados pelo plugin e aos artigos dentro da validade.
2. **Evidência:** precisa haver documentos, o melhor score precisa ser ≥ `RAG_MIN_SCORE` e a cobertura de termos precisa ser ≥ `RAG_MIN_TERM_COVERAGE`. Se falhar, a resposta é `no_evidence` **sem chamar o LLM**.
3. Contexto de até `RAG_TOP_K` trechos (no máximo 2 por artigo e `RAG_MAX_CONTEXT_CHARS`), entregues como dados delimitados.
4. O LLM responde em JSON `{"resposta", "encontrado"}` sob schema.
5. **Validação:** a resposta é descartada se `encontrado=false`, se estiver vazia, se citar artigo fora do contexto, se trouxer URL que não está no documento ou se a fundamentação ficar abaixo de `RAG_MIN_ANSWER_OVERLAP`.
6. As fontes são os artigos que de fato sustentam a resposta.

**Calibração.** O score depende do modelo de embedding e da sua KB. Monte de 20 a 50 perguntas reais:

```jsonl
{"question": "Como configuro a VPN no notebook?", "expected": [123]}
{"question": "A impressora do financeiro não imprime", "expected": [45]}
{"question": "Qual a capital da França?", "expected": []}
```

```bash
docker compose exec -T ai-api python -m app.calibrate < perguntas.jsonl
```

A saída mostra, para cada threshold, a taxa de acerto, de artigo errado e de recusa nas perguntas com resposta, e o **aceite indevido** nas perguntas sem resposta. Ela também sugere o menor threshold com aceite indevido ≤ 5%. Aplique o valor no `.env` (`RAG_MIN_SCORE`) ou na tela do plugin.

---

## API (contrato 1.0)

Todas as rotas exigem `Authorization: Bearer <AI_API_KEY>`, exceto `/api/live`.

| Método | Rota | Uso |
|---|---|---|
| GET | `/api/live` | Liveness do container |
| GET | `/api/health` | Ollama, modelos (instalado/carregado), Qdrant, acesso ao GLPI |
| GET | `/api/status` | Versão do contrato, modelos, parâmetros e estado do índice (artigos, trechos, última sincronização, erros, artigos suspeitos) |
| POST | `/api/search` | Candidatos `{article_id, title, score, url, categories}`, sem conteúdo |
| POST | `/api/chat` | `{question, allowed_article_ids, history, options}` → `{status: answered\|no_evidence\|clarify, reason, answer, sources, considered, top_score, model, timings}` |
| POST | `/api/summarize` | Título e resumo para o chamado (só com o que o usuário escreveu) |
| POST | `/api/test-llm` | Teste rápido do modelo |
| POST | `/api/index` | Reindexa artigos específicos |
| POST | `/api/sync` / `/api/reindex` | Sincronização incremental / completa em segundo plano (202) |

Os erros são sempre `{"error": "<código>"}`, sem detalhes internos: `unauthorized`, `forbidden`, `llm_unavailable`, `llm_busy`, `vector_db_unavailable`, `index_not_ready`, `sync_running`, `invalid_request`.

---

## Rede e segurança

- Publique a ai-api **só** no IP alcançável pelo GLPI (`AI_API_BIND`) e libere no firewall **só** o servidor do GLPI. Na Oracle Cloud, isso também vale para a Security List/NSG.
- Use **VPN** (WireGuard/Tailscale) ou **TLS** (`AI_API_TLS_*` ou um proxy reverso na frente) quando o tráfego passar por redes não confiáveis.
- `AI_API_ALLOWED_IPS` restringe as origens. Sem proxy, a ai-api vê o IP real de origem.
- A chave é comparada em tempo constante. Os logs são JSON com segredos mascarados. As perguntas só são registradas com `DEBUG=true`.
- Corpo máximo de 64 KB, fila de geração limitada (`OLLAMA_MAX_QUEUE`) e uma geração por vez (CPU).

---

## Testes

```bash
cd ai-api
docker run --rm -v "$PWD":/src -w /src python:3.13-slim sh -c \
  "pip install -q -r requirements-dev.txt && python -m pytest -q"
```

São 51 testes, com Qdrant em memória e substitutos determinísticos do Ollama e do GLPI. Eles cobrem RAG, threshold, injeção, artigo malicioso, permissões, sincronização (reindexação, exclusão, alteração), falhas do GLPI, do Ollama e do Qdrant, autenticação e validação. O mapa completo dos cenários está no [plugin](https://github.com/felipeaquinoti08/aisupport-/blob/main/docs/TESTES.md).

---

## Troubleshooting

| Sintoma | Verificação |
|---|---|
| `ai-api` não fica healthy | `docker compose logs ai-api`. Variável obrigatória faltando (`AI_API_KEY` < 32 caracteres, `GLPI_URL`) |
| `model-puller` falha | Saída de Internet do host. Confira `docker compose --profile setup run --rm model-puller` |
| `/api/health` com `llm.installed=false` | Rode o `model-puller`. Confira se o nome em `LLM_MODEL` é idêntico ao do `ollama list` |
| `glpi: glpi_auth_failed` | Cliente OAuth (Password, escopo api), usuário e senha |
| `glpi: glpi_unavailable` | `docker compose exec ai-api python -c "import urllib.request;print(urllib.request.urlopen('$GLPI_URL').status)"` |
| Menos artigos indexados que o esperado | A conta precisa de "Administração da base de conhecimento" |
| OOM / container reiniciando | `docker stats`. Reduza `LLM_NUM_CTX`, use um modelo menor ou aumente os limites |
| Respostas lentas | Reduza `RAG_TOP_K` / `RAG_MAX_CONTEXT_CHARS`. Mantenha `OLLAMA_KEEP_ALIVE=-1` |
| Muitas recusas | Calibre `RAG_MIN_SCORE` e enriqueça a KB (veja "Lacunas" no painel do GLPI) |

Logs: `docker compose logs -f ai-api` (JSON por linha).

---

## Backup

O índice é **derivado** da KB do GLPI e pode ser reconstruído com `python -m app.indexer.full`. O histórico das conversas fica no banco do GLPI. Ainda assim, para restaurar rápido:

```bash
# snapshot do Qdrant (fica no volume qdrant_snapshots)
docker compose exec ai-api python -c "
from app.config import get_settings; from qdrant_client import QdrantClient
s=get_settings(); c=QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key.get_secret_value())
print(c.create_snapshot(collection_name=s.qdrant_collection))"
docker compose cp qdrant:/qdrant/snapshots ./backups/
```

Guarde também o `.env`, que contém as chaves. Os modelos (volume `ollama_data`) podem ser baixados de novo.

---

## Atualização

```bash
git pull
docker compose build ai-api
docker compose up -d
```

Para atualizar o Ollama ou o Qdrant, mude `OLLAMA_IMAGE` ou `QDRANT_IMAGE` no `.env` para uma versão testada e rode `docker compose up -d`. O contrato da API é versionado: o plugin recusa um MAJOR diferente e avisa no painel.

## Remoção

```bash
docker compose down        # mantém volumes (modelos e índice)
docker compose down -v     # remove também modelos e índice
```
