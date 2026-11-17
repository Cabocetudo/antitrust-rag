"""Core domain objects shared by every layer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class LegalUnit:
    """A position inside a statute: article -> paragraph (§) -> item (inciso) -> sub-item (alínea).

    ``paragraph`` is ``None`` for text that sits in the article's *caput* (head).
    """

    article: str
    paragraph: str | None = None
    item: str | None = None
    subitem: str | None = None

    def label(self, style: str = "br") -> str:
        """Human readable locator, e.g. ``Art. 36, §3, III`` or ``Art. 101(1)(a)``."""
        if style == "eu":
            out = f"Art. {self.article}"
            if self.paragraph:
                out += f"({self.paragraph})"
            if self.item:
                out += f"({self.item})"
            return out
        parts = [f"Art. {self.article}"]
        if self.paragraph:
            parts.append("sole paragraph" if self.paragraph == "único" else f"§{self.paragraph}")
        if self.item:
            parts.append(self.item)
        if self.subitem:
            parts.append(f"{self.subitem})")
        return ", ".join(parts)


@dataclass
class Chunk:
    """A retrievable piece of a legal text with enough metadata to cite it precisely."""

    source_id: str  # e.g. "lei-12529-2011"
    source_title: str  # e.g. "Lei 12.529/2011"
    unit: LegalUnit
    text: str  # the text that is embedded/indexed (includes a context header)
    heading_path: list[str] = field(default_factory=list)  # e.g. ["TÍTULO V", "CAPÍTULO II"]
    url: str | None = None
    language: str = "pt"
    citation_style: str = "br"

    @property
    def citation(self) -> str:
        return f"{self.source_title}, {self.unit.label(self.citation_style)}"

    @property
    def chunk_id(self) -> str:
        """Stable id: same source + locator + text -> same id (makes ingestion idempotent)."""
        raw = f"{self.source_id}|{self.unit.label(self.citation_style)}|{self.text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float
    # Per-retriever ranks (1-based) — useful for debugging fusion and for the /search response.
    ranks: dict[str, int] = field(default_factory=dict)
