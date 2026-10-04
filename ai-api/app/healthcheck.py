"""Healthcheck do container: verifica apenas se o processo responde (liveness)."""

import os
import ssl
import sys
import urllib.request


def main() -> int:
    scheme = "https" if os.environ.get("AI_API_TLS_CERT") else "http"
    ctx = ssl._create_unverified_context() if scheme == "https" else None
    try:
        with urllib.request.urlopen(f"{scheme}://127.0.0.1:8080/api/live", timeout=5, context=ctx) as r:
            return 0 if r.status == 200 else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
