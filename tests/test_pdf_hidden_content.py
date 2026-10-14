import sys
from pathlib import Path

import pymupdf

sys.path.append(str(Path(__file__).parent.parent))

from redaction.pdf_redactor import PDFRedactor

SECRET = "Aditya Lawand"


def _pdf_with_hidden_pii(path: Path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Quarterly report")
    page.add_text_annot((100, 100), f"Reviewed by {SECRET}")
    page.insert_link({
        "kind": pymupdf.LINK_URI,
        "from": pymupdf.Rect(72, 60, 200, 80),
        "uri": "mailto:aditya.lawand@example.com",
    })
    doc.set_metadata({"author": f"[OC] {SECRET}", "title": "Policy", "subject": SECRET})
    doc.set_toc([[1, f"Owner {SECRET}", 1]])
    doc.embfile_add("notes.txt", f"{SECRET}".encode())
    doc.save(path)
    doc.close()


def test_redacted_pdf_has_no_metadata_comments_links_bookmarks_or_attachments(tmp_path):
    src, out = tmp_path / "in.pdf", tmp_path / "out.pdf"
    _pdf_with_hidden_pii(src)

    PDFRedactor().redact(str(src), str(out), chunks=[], pii_results=[])

    with pymupdf.open(out) as doc:
        assert not any(doc.metadata.get(k) for k in ("author", "title", "subject", "creator", "producer"))
        assert doc.get_toc() == []
        assert doc.embfile_count() == 0
        assert sum(len(list(p.annots() or [])) for p in doc) == 0
        assert sum(len(p.get_links()) for p in doc) == 0
        assert "Quarterly report" in doc[0].get_text()
    assert SECRET.encode() not in out.read_bytes()
