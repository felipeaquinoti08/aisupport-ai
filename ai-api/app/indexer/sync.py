"""Sincronização incremental: python -m app.indexer.sync"""

import sys

from ._cli import run

if __name__ == "__main__":
    sys.exit(run("sync"))
