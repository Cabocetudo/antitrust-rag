from antitrust_rag.chunking import chunk_document, chunk_segments, parse_br, parse_eu
from antitrust_rag.models import LegalUnit
from antitrust_rag.sources import SOURCES

BR_LINES = [
    "A PRESIDENTA DA REPÚBLICA",
    "TÍTULO V",
    "DAS INFRAÇÕES DA ORDEM ECONÔMICA",
    "CAPÍTULO II",
    "DAS INFRAÇÕES",
    "Art. 36. Constituem infração da ordem econômica:",
    "I - limitar a livre concorrência;",
    "§ 3º As seguintes condutas caracterizam infração da ordem econômica:",
    "I - acordar, combinar, manipular ou ajustar com concorrente, sob qualquer forma:",
    "a) os preços de bens ou serviços ofertados individualmente;",
    "d) preços, condições, vantagens ou abstenção em licitação pública;",
    "II - promover conduta comercial uniforme;",
    "§ 4º (VETADO).",
    "Art. 47-A. Texto incluído posteriormente. (Artigo acrescido pela Lei nº 14.470, de 2022)",
    "Parágrafo único. Regra única do artigo.",
    "Brasília, 30 de novembro de 2011; 190º da Independência.",
    "Art. 999. Should never be parsed (after the signature).",
]


def test_parse_br_hierarchy() -> None:
    segs = parse_br(BR_LINES)
    units = [s.unit for s in segs]
    assert units[0] == LegalUnit("36")
    assert LegalUnit("36", None, "I") in units
    assert LegalUnit("36", "3", "I", "a") in units
    assert LegalUnit("36", "3", "I", "d") in units
    assert LegalUnit("36", "3", "II") in units
    assert LegalUnit("47-A") in units
    assert LegalUnit("47-A", "único") in units


def test_parse_br_skips_vetoed_preamble_signature_and_amendment_notes() -> None:
    segs = parse_br(BR_LINES)
    texts = " ".join(s.text for s in segs)
    assert "VETADO" not in texts
    assert "PRESIDENTA" not in texts
    assert "999" not in texts
    assert "acrescido" not in texts  # "(Artigo acrescido pela Lei ...)" stripped
    assert segs[0].headings == [
        "TÍTULO V - DAS INFRAÇÕES DA ORDEM ECONÔMICA",
        "CAPÍTULO II - DAS INFRAÇÕES",
    ]


def test_legal_unit_labels() -> None:
    assert LegalUnit("36", "3", "I", "d").label() == "Art. 36, §3, I, d)"
    assert LegalUnit("87", "único").label() == "Art. 87, sole paragraph"
    assert LegalUnit("101", "3").label("eu") == "Art. 101(3)"
    assert LegalUnit("102", None, "a").label("eu") == "Art. 102(a)"


def test_parse_eu_paragraphs_and_items(sample_docs) -> None:
    doc = next(d for d in sample_docs if d.spec.source_id == "tfeu-101")
    segs = parse_eu(doc.lines)
    units = {s.unit for s in segs}
    assert LegalUnit("101", "1", "a") in units
    assert LegalUnit("101", "3", "b") in units
    assert all(s.headings and s.headings[-1].startswith("Section 1") for s in segs)


def test_short_article_stays_whole_and_long_article_splits(sample_docs) -> None:
    lei = next(d for d in sample_docs if d.spec.source_id == "lei-12529-2011")
    chunks = chunk_document(lei, max_chars=1200)
    by_article: dict[str, list] = {}
    for c in chunks:
        by_article.setdefault(c.unit.article, []).append(c)
    assert len(by_article["1"]) == 1 and by_article["1"][0].unit == LegalUnit("1")
    labels = [c.unit.label() for c in by_article["36"]]
    assert "Art. 36, §2" in labels
    assert any(lbl.startswith("Art. 36, §3, I") for lbl in labels)
    # Each split chunk carries its context header and the caput lead-in.
    para = next(c for c in by_article["36"] if c.unit.paragraph == "2")
    assert para.text.startswith("[Lei 12.529/2011 |")
    assert "Caput: Art. 36." in para.text
    assert "§ 2º Presume-se posição dominante" in para.text


def test_item_groups_repeat_paragraph_lead(sample_docs) -> None:
    lei = next(d for d in sample_docs if d.spec.source_id == "lei-12529-2011")
    chunks = [c for c in chunk_document(lei) if c.unit.article == "36" and c.unit.paragraph == "3"]
    assert len(chunks) >= 2
    assert all("§ 3º As seguintes condutas" in c.text for c in chunks)
    assert all(len(c.text) < 1800 for c in chunks)


def test_citation_format_and_stable_ids(sample_docs) -> None:
    lei = next(d for d in sample_docs if d.spec.source_id == "lei-12529-2011")
    a = chunk_document(lei)
    b = chunk_document(lei)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)
    c36 = next(c for c in a if c.unit == LegalUnit("36", "2"))
    assert c36.citation == "Lei 12.529/2011, Art. 36, §2"


def test_chunk_segments_handles_eu_without_numbered_paragraphs(sample_docs) -> None:
    doc = next(d for d in sample_docs if d.spec.source_id == "tfeu-102")
    chunks = chunk_segments(SOURCES["tfeu-102"], doc.url, parse_eu(doc.lines))
    assert len(chunks) == 1
    assert chunks[0].citation == "TFEU, Art. 102"
    assert "abuse" in chunks[0].text
