"""Text normalization shared by the lexical index, the fake embedder and the in-memory BM25."""

from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"\w+", re.UNICODE)

# Small Portuguese + English stopword list: enough to keep OR-style lexical queries focused.
_STOPWORDS_TEXT = """
a o as os um uma uns umas de do da dos das no na nos nas em por para pelo pela pelos
pelas com sem sob sobre e ou que se ao aos à às é ser sao são foi como mais menos este
esta isto esse essa isso seu sua seus suas qual quais quando onde ha há nao não deste
desta lei art the of and or to in on for by with is are be was what which who whom when
where how does do under an as at it its this that these those can may shall any from
into than then there
"""
STOPWORDS: frozenset[str] = frozenset(_STOPWORDS_TEXT.split())


def normalize_text(text: str) -> str:
    """Lowercase and strip accents ("Concorrência" -> "concorrencia")."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize_text(text))


def content_terms(text: str) -> list[str]:
    """Tokens minus stopwords and 1-char noise; used to build lexical queries."""
    return [t for t in tokenize(normalize_text(text)) if t not in STOPWORDS and len(t) > 1]
