import xml.etree.ElementTree as ET

from redaction.office_xml_utils import redact_xml, mask_text


def test_masks_pii_in_xml_text():
    xml = b"""<?xml version="1.0"?>
    <root xmlns:w="urn:test">
    <w:p><w:r><w:t>Contact alice@example.com today</w:t></w:r></w:p>
    </root>"""

    output, count = redact_xml(xml, ["alice@example.com"])

    assert count >= 1
    assert b"alice@example.com" not in output
    ET.fromstring(output)


def test_masks_pii_split_across_runs():
    xml = b"""<root xmlns:w="urn:test">
    <w:p>
        <w:r><w:t>alice@exam</w:t></w:r>
        <w:r><w:t>ple.com</w:t></w:r>
    </w:p>
    </root>"""

    output, count = redact_xml(xml, ["alice@example.com"])

    assert count >= 1
    assert b"alice@example.com" not in output
    ET.fromstring(output)


def test_masks_pii_in_xml_attributes():
    xml = b'<root description="alice@example.com">' b"<item>safe</item></root>"

    output, count = redact_xml(xml, ["alice@example.com"])

    assert count >= 1
    assert b"alice@example.com" not in output
    ET.fromstring(output)


def test_case_insensitive_replacement():
    masked, count = mask_text(
        "ALICE@EXAMPLE.COM",
        ["alice@example.com"],
    )

    assert count == 1
    assert "ALICE@EXAMPLE.COM" not in masked
