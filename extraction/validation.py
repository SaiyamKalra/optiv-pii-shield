# extraction/validation.py
"""Check that a file really is a readable PDF, DOCX or PPTX before extracting it.

The type is decided by the file's content, never by its extension alone.
"""

import logging
import zipfile
from pathlib import Path

import pymupdf

logger = logging.getLogger("Validation")

MAX_FILE_BYTES = 200 * 1024 * 1024
# A zip that inflates beyond this is treated as a zip bomb.
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 200

PDF_MAGIC = b"%PDF-"
ZIP_MAGIC = b"PK\x03\x04"
# Password-protected Office files (and legacy .doc/.ppt) are OLE compound files, not zips.
OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

OOXML_MAIN_PARTS = {
    "docx": "word/document.xml",
    "pptx": "ppt/presentation.xml",
}


class DocumentValidationError(ValueError):
    """The file can't be extracted. `reason` is a short machine-readable code."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def detect_type(path: Path) -> str:
    """Return 'pdf', 'docx' or 'pptx' from the file's content."""
    with open(path, "rb") as f:
        head = f.read(1024)

    if PDF_MAGIC in head:
        return "pdf"
    if head.startswith(OLE_MAGIC):
        raise DocumentValidationError(
            "encrypted_or_legacy",
            f"{path.name} is a password-protected Office file or a legacy .doc/.ppt",
        )
    if head.startswith(ZIP_MAGIC):
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile as exc:
            raise DocumentValidationError("corrupt", f"{path.name} is a damaged zip: {exc}") from exc
        for kind, main_part in OOXML_MAIN_PARTS.items():
            if main_part in names:
                return kind
        raise DocumentValidationError(
            "unsupported_type", f"{path.name} is a zip file but not a DOCX or PPTX"
        )
    raise DocumentValidationError(
        "unsupported_type", f"{path.name} is not a PDF, DOCX or PPTX"
    )


def _check_zip(path: Path):
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        total = sum(i.file_size for i in infos)
        compressed = sum(i.compress_size for i in infos) or 1
        if total > MAX_UNCOMPRESSED_BYTES or total / compressed > MAX_COMPRESSION_RATIO:
            raise DocumentValidationError(
                "too_large", f"{path.name} expands to {total} bytes (possible zip bomb)"
            )
        bad = zf.testzip()
        if bad is not None:
            raise DocumentValidationError("corrupt", f"{path.name}: damaged part {bad}")


def _check_pdf(path: Path):
    try:
        doc = pymupdf.open(path)
    except (RuntimeError, ValueError) as exc:
        raise DocumentValidationError("corrupt", f"{path.name} can't be opened: {exc}") from exc
    with doc:
        if not doc.is_pdf:
            raise DocumentValidationError("unsupported_type", f"{path.name} is not a PDF")
        if doc.needs_pass:
            raise DocumentValidationError("encrypted", f"{path.name} is password-protected")
        if doc.page_count == 0:
            raise DocumentValidationError("no_pages", f"{path.name} has no pages")
        if doc.is_repaired:
            logger.warning("%s was damaged and has been repaired while opening", path.name)


def validate_document(path: str | Path, max_bytes: int = MAX_FILE_BYTES) -> str:
    """Validate a file and return its detected type: 'pdf', 'docx' or 'pptx'.

    Raises DocumentValidationError with a reason of: not_found, empty, too_large,
    unsupported_type, encrypted, encrypted_or_legacy, corrupt or no_pages.
    """
    path = Path(path)
    if not path.is_file():
        raise DocumentValidationError("not_found", f"{path} does not exist")
    size = path.stat().st_size
    if size == 0:
        raise DocumentValidationError("empty", f"{path.name} is empty")
    if size > max_bytes:
        raise DocumentValidationError("too_large", f"{path.name} is {size} bytes (limit {max_bytes})")

    kind = detect_type(path)
    expected = path.suffix.lower().lstrip(".")
    if expected and expected != kind:
        logger.warning("%s looks like a %s file despite its .%s extension", path.name, kind, expected)

    if kind == "pdf":
        _check_pdf(path)
    else:
        _check_zip(path)
    return kind
