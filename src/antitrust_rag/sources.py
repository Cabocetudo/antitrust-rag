"""Public legal sources: where to download them and how to turn their HTML into clean text lines.

Every source is official and public:

* Brazil — Lei nº 12.529/2011 (competition law). Primary URL is the Presidency's Planalto portal.
  Planalto is known to refuse or redirect some non-Brazilian / datacenter IPs, so the Chamber of
  Deputies' consolidated text ("norma atualizada") is used as an automatic fallback.
* EU — TFEU Articles 101 and 102. EUR-Lex's HTML pages sit behind a bot challenge (HTTP 202), so
  the EU Publications Office "Cellar" resolver for the same CELEX numbers is used instead; it
  returns the identical EUR-Lex document without a challenge.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) antitrust-rag/0.1 "
    "(+https://github.com/; research/portfolio project)"
)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    title: str
    urls: tuple[str, ...]  # tried in order until one returns usable text
    parser: str  # "br" | "eu"
    language: str
    citation_style: str
    headers: dict[str, str] = field(default_factory=dict)
    must_contain: str = ""  # sanity marker proving we got the law, not a portal/landing page


SOURCES: dict[str, SourceSpec] = {
    "lei-12529-2011": SourceSpec(
        source_id="lei-12529-2011",
        title="Lei 12.529/2011",
        urls=(
            "https://www.planalto.gov.br/ccivil_03/_ato2011-2014/2011/lei/l12529.htm",
            "https://www2.camara.leg.br/legin/fed/lei/2011/"
            "lei-12529-30-novembro-2011-611850-normaatualizada-pl.html",
        ),
        parser="br",
        language="pt",
        citation_style="br",
        must_contain="Constituem infração da ordem econômica",
    ),
    "tfeu-101": SourceSpec(
        source_id="tfeu-101",
        title="TFEU",
        urls=("http://publications.europa.eu/resource/celex/12008E101",),
        parser="eu",
        language="en",
        citation_style="eu",
        headers={"Accept": "text/html", "Accept-Language": "eng"},
        must_contain="Article 101",
    ),
    "tfeu-102": SourceSpec(
        source_id="tfeu-102",
        title="TFEU",
        urls=("http://publications.europa.eu/resource/celex/12008E102",),
        parser="eu",
        language="en",
        citation_style="eu",
        headers={"Accept": "text/html", "Accept-Language": "eng"},
        must_contain="Article 102",
    ),
}


class SourceUnavailableError(RuntimeError):
    pass


def html_to_lines(html: str) -> list[str]:
    """Extract paragraph-level text lines from a statute HTML page.

    Struck-through text (``<strike>``, ``<s>``, ``<del>``) is dropped: Planalto uses it for
    revoked wording that is kept on the page for history.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "strike", "s", "del"]):
        tag.decompose()
    lines: list[str] = []
    for p in soup.find_all("p"):
        if p.find("p") is not None:
            continue  # some EUR-Lex pages wrap paragraphs in an outer <p>; keep leaves only
        text = " ".join(p.get_text(" ").split())
        if text:
            lines.append(text)
    return lines


def fetch_source(
    spec: SourceSpec,
    *,
    timeout_s: float = 30.0,
    attempts: int = 3,
    client: httpx.Client | None = None,
) -> tuple[str, list[str]]:
    """Download a source, trying each URL with retries. Returns ``(url_used, lines)``."""
    own_client = client is None
    client = client or httpx.Client(
        timeout=timeout_s, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )
    errors: list[str] = []
    try:
        for url in spec.urls:
            for attempt in range(1, attempts + 1):
                try:
                    resp = client.get(url, headers=spec.headers)
                    if resp.status_code != 200:
                        raise SourceUnavailableError(f"HTTP {resp.status_code}")
                    lines = html_to_lines(resp.text)
                    if spec.must_contain and not any(spec.must_contain in ln for ln in lines):
                        # e.g. Planalto redirecting to the gov.br home page
                        raise SourceUnavailableError(f"marker not found (landed on {resp.url})")
                    logger.info("fetched source", extra={"source": spec.source_id, "url": url})
                    return url, lines
                except (httpx.HTTPError, SourceUnavailableError) as exc:
                    errors.append(f"{url} attempt {attempt}: {exc}")
                    logger.warning(
                        "source fetch failed",
                        extra={"source": spec.source_id, "url": url, "error": str(exc)},
                    )
                    if isinstance(exc, SourceUnavailableError) and "marker" in str(exc):
                        break  # wrong content will not fix itself by retrying; try next URL
                    time.sleep(min(2**attempt, 8))
    finally:
        if own_client:
            client.close()
    raise SourceUnavailableError(f"{spec.source_id}: all URLs failed: {errors}")


# ---------------------------------------------------------------------------------------------
# Offline sample corpus: plain-text files with a small "# key: value" header.
# ---------------------------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^#\s*(\w+)\s*:\s*(.*)$")


@dataclass
class RawDocument:
    spec: SourceSpec
    url: str
    lines: list[str]


def write_sample(path: Path, spec: SourceSpec, url: str, lines: Iterable[str]) -> None:
    header = [
        f"# source_id: {spec.source_id}",
        f"# url: {url}",
        f"# note: {spec.title} — public legal text, excerpt for the offline demo",
    ]
    path.write_text("\n".join([*header, *lines]) + "\n", encoding="utf-8")


def read_sample(path: Path) -> RawDocument:
    meta: dict[str, str] = {}
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _HEADER_RE.match(raw)
        if m and not lines:
            meta[m.group(1)] = m.group(2).strip()
        elif raw.strip():
            lines.append(raw.strip())
    source_id = meta.get("source_id")
    if source_id not in SOURCES:
        raise ValueError(f"{path}: unknown or missing source_id {source_id!r}")
    return RawDocument(spec=SOURCES[source_id], url=meta.get("url", ""), lines=lines)


def load_sample_dir(directory: Path) -> list[RawDocument]:
    return [read_sample(p) for p in sorted(directory.glob("*.txt"))]
