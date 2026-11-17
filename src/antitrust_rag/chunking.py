"""Structure-aware parsing and chunking of statutes.

Legal text has an explicit hierarchy, and users cite it by that hierarchy, so we never split on
a fixed token window. Instead:

1. **Parse** each line into a :class:`Segment` tagged with its position
   (article → paragraph/§ → item/inciso → sub-item/alínea) plus the enclosing headings
   (TÍTULO / CAPÍTULO / Seção ...).
2. **Chunk** per article. A short article becomes one chunk. A long article is split at its
   paragraphs (caput, §1, §2 ...); a paragraph that is still too long is split into groups of
   consecutive items, repeating the paragraph's lead-in line so every chunk reads on its own.
3. Every chunk is prefixed with a small **context header** (law, heading path, locator) and, for
   paragraph-level chunks, the start of the article's caput. This "contextual header" helps both
   BM25 and embeddings, because a bare "§ 2º Presume-se..." line says nothing about its article.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from antitrust_rag.models import Chunk, LegalUnit
from antitrust_rag.sources import RawDocument, SourceSpec

ORD = r"(?:º|°|o)?"

# --- Brazilian statute patterns -------------------------------------------------------------
BR_HEADING = re.compile(r"^(TÍTULO|CAPÍTULO|Seção|Subseção|LIVRO)\s+([IVXLC]+|ÚNICO|ÚNICA)\b", re.I)
BR_ARTICLE = re.compile(rf"^Art\.\s*(\d+(?:\.\d+)?){ORD}(?:\s*-\s*([A-Z]))?\s*\.?\s*(.*)$")
BR_PARAGRAPH = re.compile(rf"^§\s*(\d+){ORD}(?:\s*-\s*([A-Z]))?\s*\.?\s*(.*)$")
BR_SOLE_PARAGRAPH = re.compile(r"^Parágrafo\s+único\.?\s*(.*)$", re.I)
BR_ITEM = re.compile(r"^([IVXLC]+)(?:\s*-\s*([A-Z]))?\s*[-–—]\s*(.*)$")
BR_SUBITEM = re.compile(r"^([a-z])\)\s*(.*)$")
BR_END = re.compile(r"^Brasília,\s")

# --- EU treaty patterns ---------------------------------------------------------------------
EU_ARTICLE = re.compile(r"^Article\s+(\d+[a-z]?)$")
EU_PARAGRAPH = re.compile(r"^(\d+)\.\s+(.*)$")
EU_ITEM = re.compile(r"^\(([a-z])\)\s+(.*)$")
EU_SKIP = re.compile(r"^(\(ex Article .*\)|-{5,}|Important legal notice|\||Official Journal.*)$")

# Amendment annotations Planalto/Câmara append to provisions, e.g.
# "(Parágrafo acrescido pela Lei nº 14.470, de 16/11/2022)" or "(Redação dada pela Lei ...)".
AMENDMENT_NOTE = re.compile(
    r"\s*\((?:[^()]*?)(?:acrescid|redação dada|incluíd|vide|vigência|produção de efeito)[^()]*\)",
    re.I,
)
DEAD_TEXT = re.compile(r"^\(?\s*(VETADO|Revogad[oa])", re.I)


@dataclass
class Segment:
    unit: LegalUnit
    text: str
    headings: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    return AMENDMENT_NOTE.sub("", text).strip()


def _article_id(num: str, suffix: str | None) -> str:
    return f"{num}-{suffix}" if suffix else num


def parse_br(lines: Iterable[str]) -> list[Segment]:
    """Parse a Brazilian statute (Planalto / Câmara layout) into hierarchical segments."""
    segments: list[Segment] = []
    headings: dict[str, str] = {}  # level -> "CAPÍTULO II - DAS INFRAÇÕES"
    order = ["LIVRO", "TÍTULO", "CAPÍTULO", "SEÇÃO", "SUBSEÇÃO"]
    pending_heading: str | None = None
    article: str | None = None
    paragraph: str | None = None
    item: str | None = None

    def heading_path() -> list[str]:
        return [headings[k] for k in order if k in headings]

    def add(unit: LegalUnit, text: str) -> None:
        text = _clean(text)
        body = text.split(" ", 2)[-1] if text else ""
        if not text or DEAD_TEXT.match(body) or DEAD_TEXT.match(text):
            return
        segments.append(Segment(unit=unit, text=text, headings=heading_path()))

    for raw in lines:
        line = " ".join(raw.split())
        if not line:
            continue
        if BR_END.match(line) and article is not None:
            break
        if m := BR_HEADING.match(line):
            level = m.group(1).upper()
            headings[level] = line
            for deeper in order[order.index(level) + 1 :]:
                headings.pop(deeper, None)
            pending_heading = level
            continue
        if pending_heading and not BR_ARTICLE.match(line):
            # The line after "CAPÍTULO II" is its name ("DAS INFRAÇÕES").
            headings[pending_heading] = f"{headings[pending_heading]} - {line}"
            pending_heading = None
            continue
        pending_heading = None

        if m := BR_ARTICLE.match(line):
            article, paragraph, item = _article_id(m.group(1), m.group(2)), None, None
            add(LegalUnit(article), line)
        elif article is None:
            continue  # preamble ("A PRESIDENTA DA REPÚBLICA ...")
        elif m := BR_SOLE_PARAGRAPH.match(line):
            paragraph, item = "único", None
            add(LegalUnit(article, paragraph), line)
        elif m := BR_PARAGRAPH.match(line):
            paragraph, item = _article_id(m.group(1), m.group(2)), None
            add(LegalUnit(article, paragraph), line)
        elif m := BR_ITEM.match(line):
            item = _article_id(m.group(1), m.group(2))
            add(LegalUnit(article, paragraph, item), line)
        elif m := BR_SUBITEM.match(line):
            add(LegalUnit(article, paragraph, item, m.group(1)), line)
        elif segments:
            segments[-1].text += " " + _clean(line)  # wrapped continuation line
    return segments


def parse_eu(lines: Iterable[str]) -> list[Segment]:
    """Parse an EUR-Lex treaty article page (TFEU layout) into hierarchical segments."""
    segments: list[Segment] = []
    heading: list[str] = []
    article: str | None = None
    paragraph: str | None = None
    for raw in lines:
        line = " ".join(raw.split())
        if not line or EU_SKIP.match(line):
            continue
        if line.startswith("Consolidated version of the Treaty"):
            # "... - TITLE VII: ... - Chapter 1: Rules on competition - Section 1: ... - Article 1"
            parts = [p.strip() for p in line.split(" - ")]
            heading = [p for p in parts if re.match(r"^(PART|TITLE|Chapter|Section)\b", p)]
            continue
        if m := EU_ARTICLE.match(line):
            article, paragraph = m.group(1), None
            continue
        if article is None:
            continue
        if m := EU_PARAGRAPH.match(line):
            paragraph = m.group(1)
            segments.append(Segment(LegalUnit(article, paragraph), line, list(heading)))
        elif m := EU_ITEM.match(line):
            segments.append(Segment(LegalUnit(article, paragraph, m.group(1)), line, list(heading)))
        else:
            # Unnumbered text: caput (Art. 102) or dash-lists / continuations (Art. 101(3)).
            segments.append(Segment(LegalUnit(article, paragraph), line, list(heading)))
    return segments


PARSERS = {"br": parse_br, "eu": parse_eu}


# ---------------------------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------------------------


def _header(spec: SourceSpec, unit: LegalUnit, headings: list[str]) -> str:
    path = " > ".join(headings[-2:])  # the two innermost headings are the informative ones
    loc = unit.label(spec.citation_style)
    return f"[{spec.title} | {path} | {loc}]" if path else f"[{spec.title} | {loc}]"


def _make_chunk(
    spec: SourceSpec,
    url: str,
    unit: LegalUnit,
    segs: list[Segment],
    lead: str = "",
) -> Chunk:
    body = "\n".join(s.text for s in segs)
    text = _header(spec, unit, segs[0].headings) + "\n" + (f"{lead}\n" if lead else "") + body
    return Chunk(
        source_id=spec.source_id,
        source_title=spec.title,
        unit=unit,
        text=text,
        heading_path=list(segs[0].headings),
        url=url,
        language=spec.language,
        citation_style=spec.citation_style,
    )


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def chunk_segments(
    spec: SourceSpec,
    url: str,
    segments: list[Segment],
    *,
    max_chars: int = 1200,
    lead_chars: int = 220,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    by_article: dict[str, list[Segment]] = {}
    for seg in segments:
        by_article.setdefault(seg.unit.article, []).append(seg)

    for article, segs in by_article.items():
        total = sum(len(s.text) + 1 for s in segs)
        if total <= max_chars:
            chunks.append(_make_chunk(spec, url, LegalUnit(article), segs))
            continue

        has_caput = segs[0].unit.paragraph is None
        caput_lead = "Caput: " + _truncate(segs[0].text, lead_chars) if has_caput else ""
        groups: dict[str | None, list[Segment]] = {}
        for seg in segs:
            groups.setdefault(seg.unit.paragraph, []).append(seg)

        for paragraph, psegs in groups.items():
            lead = "" if paragraph is None else caput_lead
            unit = LegalUnit(article, paragraph)
            if sum(len(s.text) + 1 for s in psegs) <= max_chars:
                chunks.append(_make_chunk(spec, url, unit, psegs, lead))
                continue
            # Paragraph still too long: keep its lead-in line, pack items greedily.
            intro, rest = psegs[0], psegs[1:]
            p_lead = (lead + "\n" if lead else "") + _truncate(intro.text, lead_chars)
            if not rest:
                chunks.append(_make_chunk(spec, url, unit, psegs, lead))
                continue
            batch: list[Segment] = []
            size = 0
            for seg in rest:
                if batch and size + len(seg.text) > max_chars - len(p_lead):
                    chunks.append(_item_chunk(spec, url, article, paragraph, batch, p_lead))
                    batch, size = [], 0
                batch.append(seg)
                size += len(seg.text) + 1
            if batch:
                chunks.append(_item_chunk(spec, url, article, paragraph, batch, p_lead))
    return chunks


def _item_chunk(
    spec: SourceSpec,
    url: str,
    article: str,
    paragraph: str | None,
    batch: list[Segment],
    lead: str,
) -> Chunk:
    items = [s.unit.item for s in batch if s.unit.item]
    first, last = (items[0], items[-1]) if items else (None, None)
    item_label = first if first == last else f"{first}-{last}"
    return _make_chunk(spec, url, LegalUnit(article, paragraph, item_label), batch, lead)


def chunk_document(doc: RawDocument, *, max_chars: int = 1200) -> list[Chunk]:
    segments = PARSERS[doc.spec.parser](doc.lines)
    return chunk_segments(doc.spec, doc.url, segments, max_chars=max_chars)
