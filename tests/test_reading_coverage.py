from analytics.metrics import reading_coverage
from models.schemas import ExtractionInput, UnreadRegion


def _chunk(text="", **fields):
    return ExtractionInput(file_id="f", file_name="f.pdf", raw_text=text, **fields)


def test_clean_native_document_is_fully_covered():
    result = reading_coverage([_chunk("one two three four")])
    assert result == {"total": 4, "reliable": 4, "coverage_pct": 100.0}


def test_low_confidence_ocr_words_lower_coverage():
    result = reading_coverage([_chunk("a b c d e f g h i j", low_confidence_word_count=2)])
    assert result["coverage_pct"] == 80.0


def test_unread_region_counts_against_coverage():
    region = UnreadRegion(page=1, x=0, y=0, w=10, h=10)
    result = reading_coverage([_chunk("a b c", unread_regions=[region])])
    assert result["total"] == 4
    assert result["reliable"] == 3


def test_page_with_extraction_errors_is_not_reliable():
    result = reading_coverage(
        [_chunk("a b c d"), _chunk("e f g h", extraction_errors=["page extraction failed"])]
    )
    assert result["coverage_pct"] == 50.0


def test_blank_page_with_errors_still_counts():
    result = reading_coverage([_chunk("", extraction_errors=["page extraction failed"])])
    assert result["total"] == 1
    assert result["coverage_pct"] == 0.0


def test_empty_document_is_not_flagged():
    assert reading_coverage([])["coverage_pct"] == 100.0
