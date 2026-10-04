#!/bin/sh
# Executado dentro do serviço "model-puller" (docker compose --profile setup).
# Sobe um Ollama temporário com acesso à Internet, baixa os modelos para o
# volume compartilhado e encerra.
set -eu

ollama serve >/tmp/ollama.log 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true' EXIT

i=0
until ollama list >/dev/null 2>&1; do
    i=$((i + 1))
    if [ "$i" -gt 30 ]; then
        echo "Ollama temporário não respondeu:" >&2
        cat /tmp/ollama.log >&2
        exit 1
    fi
    sleep 1
done

for model in "$LLM_MODEL" "$EMBED_MODEL" $EXTRA_MODELS; do
    echo ">> Baixando $model"
    ollama pull "$model"
done

echo ">> Modelos disponíveis:"
ollama list
