# Sample corpus (offline)

Small excerpts of public legal texts so the demo, tests and evaluation run without network access.
Each `.txt` file starts with a `# key: value` header (`source_id`, `url`) followed by one
paragraph of the official text per line, exactly as extracted by `antitrust_rag.sources.html_to_lines`.

| File | Content | Downloaded from |
|---|---|---|
| `lei_12529_2011_excerpt.txt` | Lei nº 12.529/2011 (Brazil), Arts. 1, 2, 31, 36, 37, 38, 45, 47, 86, 87, 88, 90 with their headings | Câmara dos Deputados consolidated text ("norma atualizada") |
| `tfeu_art_101.txt` | TFEU Article 101 | EU Publications Office (Cellar), CELEX 12008E101 |
| `tfeu_art_102.txt` | TFEU Article 102 | EU Publications Office (Cellar), CELEX 12008E102 |

Run `python scripts/ingest.py --origin remote` to index the full law instead of the excerpt.

**Reuse.** Brazilian statutes are not protected by copyright (Lei nº 9.610/1998, Art. 8, IV).
EU legal texts from EUR-Lex may be reused with acknowledgement of the source
(Commission Decision 2011/833/EU). Source: © European Union, https://eur-lex.europa.eu.
These excerpts are for demonstration only; the official publications prevail.
