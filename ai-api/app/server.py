"""Ponto de entrada: inicia o uvicorn, com TLS quando configurado."""

import os

import uvicorn


def main() -> None:
    cert = os.environ.get("AI_API_TLS_CERT") or None
    key = os.environ.get("AI_API_TLS_KEY") or None
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        workers=1,
        proxy_headers=False,
        server_header=False,
        log_level=os.environ.get("LOG_LEVEL", "INFO").lower(),
        ssl_certfile=cert,
        ssl_keyfile=key,
    )


if __name__ == "__main__":
    main()
