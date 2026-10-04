"""Camada 2 de proteção: decide, no backend, se existe evidência suficiente.

O LLM só é chamado quando há documentos relevantes acima do threshold. Depois,
a resposta precisa vir marcada como encontrada, não citar artigos fora do
contexto, não trazer URLs externas e ser lexicalmente fundamentada nos
documentos; caso contrário é descartada.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .text import answer_overlap, fold, term_coverage, tokens
from .vectorstore import Hit


@dataclass
class RagParams:
    min_score: float
    min_term_coverage: float
    min_answer_overlap: float
    context_margin: float
    top_k: int
    max_chunks_per_article: int
    max_context_chars: int


@dataclass
class EvidenceDecision:
    ok: bool
    reason: str
    top_score: float
    coverage: float
    context: list[Hit] = field(default_factory=list)


def evaluate_evidence(question: str, hits: list[Hit], p: RagParams) -> EvidenceDecision:
    if not tokens(question):
        return EvidenceDecision(False, "empty_query", 0.0, 0.0)
    if not hits:
        return EvidenceDecision(False, "no_documents", 0.0, 0.0)

    top_score = max(h.dense_score for h in hits)
    if top_score < p.min_score:
        return EvidenceDecision(False, "low_score", top_score, 0.0)

    floor = max(p.min_score - p.context_margin, 0.0)
    selected: list[Hit] = []
    per_article: dict[int, int] = {}
    used = 0
    for h in hits:  # já ordenados pela fusão densa+esparsa
        if h.dense_score < floor:
            continue
        if per_article.get(h.article_id, 0) >= p.max_chunks_per_article:
            continue
        size = len(h.text) + len(h.title)
        if selected and used + size > p.max_context_chars:
            continue
        selected.append(h)
        per_article[h.article_id] = per_article.get(h.article_id, 0) + 1
        used += size
        if len(selected) >= p.top_k:
            break

    if not selected:
        return EvidenceDecision(False, "low_score", top_score, 0.0)

    context_text = "\n".join(f"{h.title}\n{h.text}" for h in selected)
    coverage = term_coverage(question, context_text)
    if coverage < p.min_term_coverage:
        return EvidenceDecision(False, "low_coverage", top_score, coverage)
    return EvidenceDecision(True, "ok", top_score, coverage, selected)


_CITATION = re.compile(r"\[?\s*KB\s*#?\s*(\d+)\s*\]?", re.I)
_URL = re.compile(r"https?://[^\s)\]>\"']+", re.I)
_MIN_ANSWER_TOKENS = 3


@dataclass
class AnswerCheck:
    ok: bool
    reason: str
    answer: str
    source_ids: list[int]
    overlap: float


def parse_llm_output(raw: str) -> tuple[str, bool] | None:
    """Lê o JSON {"resposta": str, "encontrado": bool} gerado sob schema."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("encontrado"), bool):
        return None
    answer = data.get("resposta")
    return (answer if isinstance(answer, str) else ""), data["encontrado"]


def validate_answer(raw: str, context: list[Hit], p: RagParams) -> AnswerCheck:
    parsed = parse_llm_output(raw)
    if parsed is None:
        return AnswerCheck(False, "invalid_output", "", [], 0.0)
    answer, found = parsed
    answer = answer.strip()
    if not found:
        return AnswerCheck(False, "llm_declined", "", [], 0.0)

    allowed = {h.article_id for h in context}
    cited = list(dict.fromkeys(int(m.group(1)) for m in _CITATION.finditer(answer)))
    if any(aid not in allowed for aid in cited):
        return AnswerCheck(False, "invalid_citation", "", cited, 0.0)
    answer = re.sub(r"[ \t]+\n", "\n", _CITATION.sub("", answer)).strip()

    if len(tokens(answer)) < _MIN_ANSWER_TOKENS:
        return AnswerCheck(False, "empty_answer", "", [], 0.0)

    context_text = "\n".join(f"{h.title}\n{h.text}" for h in context)
    folded_context = fold(context_text)
    for url in _URL.findall(answer):
        if fold(url.rstrip(".,;")) not in folded_context:
            return AnswerCheck(False, "external_url", "", [], 0.0)

    overlap = answer_overlap(answer, context_text)
    if overlap < p.min_answer_overlap:
        return AnswerCheck(False, "low_grounding", "", [], overlap)
    return AnswerCheck(True, "ok", answer, attribute_sources(answer, context, cited), overlap)


def attribute_sources(answer: str, context: list[Hit], cited: list[int]) -> list[int]:
    """Artigos que de fato sustentam a resposta (sobreposição lexical), mais os citados."""
    texts: dict[int, str] = {}
    for h in context:
        texts[h.article_id] = texts.get(h.article_id, "") + f"\n{h.title}\n{h.text}"
    scores = {aid: answer_overlap(answer, text) for aid, text in texts.items()}
    best = max(scores.values(), default=0.0)
    chosen = [aid for aid, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True)
              if sc >= max(0.3, best * 0.6)]
    for aid in cited:
        if aid not in chosen:
            chosen.append(aid)
    return chosen
