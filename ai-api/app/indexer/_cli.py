"""Execução da indexação pela linha de comando (docker compose exec ai-api ...)."""

from __future__ import annotations

import asyncio
import json
import sys

from ..config import get_settings
from ..container import build_container
from ..logging_setup import setup_logging


def run(mode: str) -> int:
    settings = get_settings()
    setup_logging(settings.log_level, [])

    async def _main() -> int:
        c = build_container(settings)
        try:
            result = await (c.indexer.full() if mode == "full" else c.indexer.sync())
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except Exception as e:
            await c.indexer.record_error(mode, e)
            print(f"Falha na indexação ({mode}): {getattr(e, 'code', type(e).__name__)}", file=sys.stderr)
            return 1
        finally:
            await c.aclose()

    return asyncio.run(_main())
