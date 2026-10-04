#!/bin/sh
# Executado dentro do serviço "model-puller" (docker compose --profile setup,
# ou automaticamente a cada deploy com COMPOSE_PROFILES=setup - ex.: Komodo).
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

# Idempotente: modelos já presentes no volume não são baixados de novo
# (o serviço pode rodar a cada deploy, inclusive sem Internet).
installed=$(ollama list | awk 'NR>1 {print $1}')
for model in "$LLM_MODEL" "$EMBED_MODEL" $EXTRA_MODELS; do
    case "$model" in *:*) name="$model" ;; *) name="$model:latest" ;; esac
    if printf '%s\n' "$installed" | grep -qx "$name"; then
        echo ">> $model já está instalado"
        continue
    fi
    echo ">> Baixando $model"
    ollama pull "$model"
done

echo ">> Modelos disponíveis:"
ollama list
