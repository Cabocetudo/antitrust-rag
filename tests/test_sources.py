from pathlib import Path

import httpx
import pytest

from antitrust_rag.sources import (
    SOURCES,
    SourceUnavailableError,
    fetch_source,
    html_to_lines,
    read_sample,
    write_sample,
)

LAW_HTML = """<html><body>
<p>Art. 36. Constituem infração da ordem econômica, independentemente de culpa:</p>
<p><strike>Texto revogado que não deve aparecer</strike></p>
<p>I - limitar a concorrência;</p>
<p>Outer <p>Nested leaf</p></p>
<script>var x = 1;</script>
</body></html>"""

PORTAL_HTML = "<html><body><p>Página Inicial — Planalto</p></body></html>"


def test_html_to_lines_drops_struck_text_and_scripts() -> None:
    lines = html_to_lines(LAW_HTML)
    assert lines[0].startswith("Art. 36.")
    assert not any("revogado" in ln for ln in lines)
    assert not any("var x" in ln for ln in lines)
    assert "Nested leaf" in lines


def test_fetch_falls_back_when_primary_lands_on_portal(monkeypatch) -> None:
    monkeypatch.setattr("antitrust_rag.sources.time.sleep", lambda s: None)
    spec = SOURCES["lei-12529-2011"]
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if "planalto" in request.url.host:
            return httpx.Response(200, text=PORTAL_HTML)  # geo-redirected landing page
        return httpx.Response(200, text=LAW_HTML)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        url, lines = fetch_source(spec, client=client)
    assert "camara.leg.br" in url
    assert any("Constituem infração" in ln for ln in lines)
    assert len(seen) == 2  # wrong content is not retried; we move on to the fallback


def test_fetch_raises_after_all_urls_fail(monkeypatch) -> None:
    monkeypatch.setattr("antitrust_rag.sources.time.sleep", lambda s: None)
    transport = httpx.MockTransport(lambda r: httpx.Response(503))
    with (
        httpx.Client(transport=transport) as client,
        pytest.raises(SourceUnavailableError),
    ):
        fetch_source(SOURCES["tfeu-101"], client=client, attempts=2)


def test_sample_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    write_sample(p, SOURCES["tfeu-102"], "http://example/102", ["Article 102", "Any abuse ..."])
    doc = read_sample(p)
    assert doc.spec.source_id == "tfeu-102"
    assert doc.url == "http://example/102"
    assert doc.lines == ["Article 102", "Any abuse ..."]


def test_bundled_sample_corpus_is_complete(sample_docs) -> None:
    ids = {d.spec.source_id for d in sample_docs}
    assert ids == {"lei-12529-2011", "tfeu-101", "tfeu-102"}
