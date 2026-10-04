"""Reindexação completa: python -m app.indexer.full"""

import sys

from ._cli import run

if __name__ == "__main__":
    sys.exit(run("full"))
