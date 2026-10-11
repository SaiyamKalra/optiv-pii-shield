from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re
import xml.etree.ElementTree as ET


def get_pii_values(pii_results):
    values = set()

    for result in pii_results:
        for entity in result.detected_entities:
            value = getattr(entity, "original_value", None)
            if value:
                value = str(value).strip()
                if value:
                    values.add(value)

    return sorted(values, key=len, reverse=True)


def mask_text(text, values):
    if not text:
        return text, 0

    count = 0

    for value in values:
        pattern = re.compile(re.escape(value), re.IGNORECASE)

        def replace(match):
            nonlocal count
            count += 1
            return "\u2588" * len(match.group())

        text = pattern.sub(replace, text)

    return text, count


def redact_xml(data, values):
    root = ET.fromstring(data)
    count = 0

    # First handle PII split across multiple Word/PowerPoint text runs.
    paragraphs = [node for node in root.iter() if node.tag.split("}")[-1] in ("p",)]

    for paragraph in paragraphs:
        text_nodes = [
            node
            for node in paragraph.iter()
            if node.tag.split("}")[-1] == "t" and node.text
        ]

        if not text_nodes:
            continue

        combined = "".join(node.text for node in text_nodes)
        masked, replacements = mask_text(combined, values)

        if replacements:
            count += replacements
            position = 0

            for node in text_nodes:
                length = len(node.text)
                node.text = masked[position : position + length]
                position += length

    # Inspect every remaining XML text node, not only <w:t> nodes.
    # This includes many metadata, notes, comments and property fields.
    for node in root.iter():
        if node.text:
            node.text, replacements = mask_text(node.text, values)
            count += replacements

        # PII can also be stored in XML attributes, including alt text.
        for key, value in list(node.attrib.items()):
            node.attrib[key], replacements = mask_text(value, values)
            count += replacements

        if node.tail:
            node.tail, replacements = mask_text(node.tail, values)
            count += replacements

    return (
        ET.tostring(
            root,
            encoding="utf-8",
            xml_declaration=True,
        ),
        count,
    )


def redact_office_archive(input_path, output_path, pii_results):
    source = Path(input_path)
    destination = Path(output_path)

    if source.resolve() == destination.resolve():
        raise ValueError("Output must be different from the original file.")

    if not source.is_file():
        raise FileNotFoundError(source)

    values = get_pii_values(pii_results)

    if not values:
        raise ValueError("No detected PII values were supplied.")

    total = 0

    try:
        with ZipFile(source, "r") as original:
            # Reject corrupt archives and duplicate part names.
            bad_part = original.testzip()
            if bad_part:
                raise ValueError(f"Corrupt Office package part: {bad_part}")

            names = original.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Office package contains duplicate part names.")

            with ZipFile(destination, "w", ZIP_DEFLATED) as cleaned:
                for item in original.infolist():
                    data = original.read(item.filename)

                    # Process XML and relationship files only.
                    # Binary media is deliberately not treated as XML.
                    if item.filename.endswith(".xml") or item.filename.endswith(
                        ".rels"
                    ):
                        try:
                            data, replacements = redact_xml(data, values)
                            total += replacements
                        except (ET.ParseError, UnicodeDecodeError) as exc:
                            raise ValueError(
                                f"Could not inspect Office part: {item.filename}"
                            ) from exc

                    cleaned.writestr(item, data)

        # Re-open and validate the output archive.
        with ZipFile(destination, "r") as check:
            bad_part = check.testzip()
            if bad_part:
                raise ValueError(f"Output package is corrupt: {bad_part}")

            for name in check.namelist():
                if name.endswith(".xml") or name.endswith(".rels"):
                    try:
                        ET.fromstring(check.read(name))
                    except ET.ParseError as exc:
                        raise ValueError(f"Output XML is invalid: {name}") from exc

        if total == 0:
            raise ValueError(
                "No PII text was replaced; refusing to produce verified output."
            )

        return total

    except Exception:
        destination.unlink(missing_ok=True)
        raise
