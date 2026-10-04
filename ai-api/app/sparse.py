"""Vetores esparsos estilo BM25 para a busca por palavra-chave no Qdrant.

O Qdrant aplica o IDF (Modifier.IDF) do lado do servidor; aqui calculamos só a
parte de frequência do termo (TF saturado com normalização de tamanho).
Implementação local e sem modelos externos, para funcionar offline.
"""

from __future__ import annotations

import zlib
from collections import Counter

from .text import stem, tokens

_K1 = 1.2
_B = 0.75
_AVG_LEN = 120.0


def _index(term: str) -> int:
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


def _terms(text: str) -> list[str]:
    return [stem(t) for t in tokens(text)]


def document_vector(text: str) -> tuple[list[int], list[float]]:
    terms = _terms(text)
    if not terms:
        return [], []
    counts = Counter(terms)
    length_norm = 1 - _B + _B * (len(terms) / _AVG_LEN)
    merged: dict[int, float] = {}
    for term, tf in counts.items():
        idx = _index(term)
        merged[idx] = merged.get(idx, 0.0) + tf * (_K1 + 1) / (tf + _K1 * length_norm)
    return list(merged.keys()), list(merged.values())


def query_vector(text: str) -> tuple[list[int], list[float]]:
    merged: dict[int, float] = {}
    for term in set(_terms(text)):
        merged[_index(term)] = 1.0
    return list(merged.keys()), list(merged.values())
