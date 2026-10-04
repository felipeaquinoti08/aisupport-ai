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
| ollama | 3200m | ~2,9 GB |
| qdrant | 448m | ~25 MB (KB vazia) |
| ai-api | 384m | ~40 MB |
