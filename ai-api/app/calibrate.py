"""Calibração do RAG_MIN_SCORE com perguntas reais da empresa.

Entrada (stdin), uma pergunta por linha em JSON:
  {"question": "Como configuro a VPN no notebook?", "expected": [123]}
  {"question": "Qual a capital da França?", "expected": []}     # sem resposta na KB

Uso:
  docker compose exec -T ai-api python -m app.calibrate < perguntas.jsonl

Para cada threshold mostra, nas perguntas COM resposta, quantas seriam aceitas
com o artigo certo / com artigo errado / recusadas; e, nas perguntas SEM
resposta, quantas seriam aceitas indevidamente. Sugere o menor threshold que
mantém o aceite indevido abaixo do limite (--max-false-accept, padrão 5%).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import get_settings
from .container import build_container
from .logging_setup import setup_logging


async def _collect(questions: list[dict]) -> list[dict]:
    c = build_container(get_settings())
    rows = []
    try:
        for q in questions:
            hits = await c.rag.retrieve(q["question"], 10)
            top = max(hits, key=lambda h: h.dense_score) if hits else None
            rows.append({
                "question": q["question"],
                "expected": set(int(i) for i in q.get("expected") or []),
                "top_id": top.article_id if top else None,
                "top_score": top.dense_score if top else 0.0,
            })
    finally:
        await c.aclose()
    return rows


def _report(rows: list[dict], max_false_accept: float) -> None:
    answerable = [r for r in rows if r["expected"]]
    unanswerable = [r for r in rows if not r["expected"]]
    print(f"Perguntas: {len(rows)} (com resposta: {len(answerable)}, sem resposta: {len(unanswerable)})\n")
    print(" score | certo | errado | recusa | aceite indevido")
    print("-------+-------+--------+--------+----------------")
    suggestion = None
    t = 0.30
    while t <= 0.901:
        right = sum(1 for r in answerable if r["top_score"] >= t and r["top_id"] in r["expected"])
        wrong = sum(1 for r in answerable if r["top_score"] >= t and r["top_id"] not in r["expected"])
        refused = len(answerable) - right - wrong
        false_accept = sum(1 for r in unanswerable if r["top_score"] >= t)
        fa_rate = false_accept / len(unanswerable) if unanswerable else 0.0
        pct = lambda n, d: f"{(100 * n / d):5.1f}%" if d else "   - "
        print(f" {t:.3f} | {pct(right, len(answerable))} | {pct(wrong, len(answerable))} | {pct(refused, len(answerable))} | {pct(false_accept, len(unanswerable))}")
        if suggestion is None and fa_rate <= max_false_accept:
            suggestion = t
        t = round(t + 0.025, 3)
    print()
    if suggestion is not None:
        print(f"Sugestão: RAG_MIN_SCORE={suggestion:.3f} (aceite indevido <= {max_false_accept:.0%}).")
    else:
        print("Nenhum threshold atende ao limite de aceite indevido; revise as perguntas ou a KB.")
    print("\nDetalhe (score do melhor artigo):")
    for r in sorted(rows, key=lambda r: r["top_score"], reverse=True):
        mark = "SEM" if not r["expected"] else ("ok " if r["top_id"] in r["expected"] else "ERR")
        print(f"  [{mark}] {r['top_score']:.3f}  KB#{r['top_id']}  {r['question'][:70]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-false-accept", type=float, default=0.05)
    args = parser.parse_args()
    setup_logging("WARNING", [])
    questions = []
    for n, line in enumerate(sys.stdin, 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            if not isinstance(item.get("question"), str):
                raise ValueError
        except ValueError:
            print(f"Linha {n} inválida: esperado JSON com 'question'", file=sys.stderr)
            return 2
        questions.append(item)
    if not questions:
        print("Nenhuma pergunta recebida no stdin.", file=sys.stderr)
        return 2
    _report(asyncio.run(_collect(questions)), args.max_false_accept)
    return 0


if __name__ == "__main__":
    sys.exit(main())
