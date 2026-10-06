"""Tratamento de texto: HTML -> texto, normalização, tokens, trechos e detecção de injeção."""

from __future__ import annotations

import html
import re
import unicodedata
from html.parser import HTMLParser

_BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6",
    "pre", "blockquote", "section", "article", "hr", "dt", "dd",
}
_SKIP_TAGS = {"script", "style", "head", "title", "noscript", "iframe", "object", "svg"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag, attrs):
        if tag in ("br", "hr"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def html_to_text(content: str | None) -> str:
    """Converte o HTML de um artigo do GLPI em texto simples, preservando parágrafos e listas."""
    if not content:
        return ""
    # O GLPI pode devolver o HTML com entidades codificadas (&lt;p&gt;).
    if "&lt;" in content and "<" not in content:
        content = html.unescape(content)
    parser = _TextExtractor()
    parser.feed(content)
    parser.close()
    text = "".join(parser.parts).replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_input(text: str) -> str:
    """Remove caracteres de controle e normaliza espaços de uma entrada do usuário."""
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def fold(text: str) -> str:
    """Minúsculas sem acentos, para comparação lexical."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


STOPWORDS = frozenset(
    fold(w)
    for w in """
    a à ao aos as às o os um uma uns umas de da das do dos dum duma em na nas no nos num numa
    por pela pelas pelo pelos para pra pro com sem sob sobre entre até após ante desde contra
    e ou mas nem que se porque pois como quando onde quem qual quais cujo cuja
    eu tu ele ela nós vós eles elas me te lhe nos vos lhes meu minha meus minhas teu tua seu sua
    seus suas nosso nossa isso isto aquilo esse essa este esta aquele aquela esses essas estes
    estas aqueles aquelas
    é são ser foi era estar está estou estão esta tem têm ter tenho tinha há havia vai vou fazer
    faz fiz pode posso consigo conseguir preciso precisa quero queria gostaria
    não sim já também muito mais menos mesmo ainda só apenas bem mal lá aqui ali agora então
    olá ola oi bom dia boa tarde noite obrigado obrigada favor ajuda ajudar problema problemas
    funciona funcionando funcionou funcionar consegue conseguindo consegui conseguir dando deu
    aparece aparecendo acontece acontecendo ocorre ocorrendo estava estou estamos fica ficou
    continua continuando sempre hoje ontem agora alguém alguem pessoal coisa algo ninguem ninguém
    the of and to in is it for on with
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")


def tokens(text: str) -> list[str]:
    """Tokens significativos (sem acento, minúsculos, sem stopwords)."""
    return [t for t in _WORD.findall(fold(text)) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())]


def stem(token: str) -> str:
    """Radical grosseiro (prefixo) para tolerar flexões do português."""
    return token if len(token) <= 5 or token.isdigit() else token[:5]


def term_coverage(question: str, context: str) -> float:
    """Fração dos termos significativos da pergunta que aparecem no contexto."""
    q = {stem(t) for t in tokens(question)}
    if not q:
        return 0.0
    c = {stem(t) for t in tokens(context)}
    return len(q & c) / len(q)


def answer_overlap(answer: str, context: str) -> float:
    """Fração dos termos da resposta presentes no contexto (fundamentação lexical)."""
    a = [stem(t) for t in tokens(answer) if not re.fullmatch(r"kb|\d+", t)]
    if not a:
        return 1.0
    c = {stem(t) for t in tokens(context)}
    return sum(1 for t in a if t in c) / len(a)


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Divide o texto em trechos de até `size` caracteres respeitando parágrafos."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    pieces: list[str] = []
    for p in paragraphs:
        while len(p) > size:
            cut = p.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            pieces.append(p)

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        tail = current[-overlap:] if overlap else ""
        if tail and " " in tail:
            tail = tail[tail.find(" ") + 1:]
        current = f"{tail}\n{piece}" if tail and len(tail) + len(piece) + 1 <= size else piece
    if current:
        chunks.append(current)
    return chunks


# Padrões típicos de instruções embutidas em documentos (prompt injection indireta).
_INJECTION_PATTERNS = [
    r"ignor[ea]\w*\s+(?:\w+\s+){0,3}(?:instru|regra|orienta|prompt|comando)",
    r"ignore\s+(?:\w+\s+){0,3}(?:instructions?|rules?|prompt)",
    r"desconsider\w*\s+(?:\w+\s+){0,3}(?:instru|regra|orienta)",
    r"disregard\s+(?:\w+\s+){0,3}(?:instructions?|rules?)",
    r"(?:system|sistema)\s*prompt",
    r"(?:a\s+partir\s+de\s+agora|agora)\s+voce\s+(?:e|sera|deve|vai)\b",
    r"voce\s+agora\s+(?:e|sera|deve)\b",
    r"you\s+are\s+now\b",
    r"(?:novas?|new)\s+(?:instru\w*|instructions?)\s*:",
    r"<\s*/?\s*(?:system|assistant|documento|instruc\w*)\s*>",
    r"\[/?inst\]|<\|im_start\|>|<\|im_end\|>",
    r"responda\s+(?:sempre\s+)?(?:que|com)\s+.{0,40}(?:independente|mesmo\s+que)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS))


def looks_like_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(fold(text)))


def neutralize_for_prompt(text: str) -> str:
    """Impede que o conteúdo feche/abra os delimitadores usados no prompt."""
    text = re.sub(r"<\s*/?\s*(documento|documentos|pergunta|conversa|solicitacao|historico|instrucoes_admin|escopo|system|assistant|user)\b[^>]*>", "[tag removida]", text, flags=re.I)
    text = text.replace("<|", "< |").replace("|>", "| >")
    return text
