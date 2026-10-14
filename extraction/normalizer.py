import hashlib
import io
import json
import logging
import re
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

from models.schemas import BoundingBox, ExtractionInput, HiddenContent, UnreadRegion
from ocr.ocr_engine import OCR_WORKERS, OcrEngine, OcrResult, OcrWord, is_meaningful

logger = logging.getLogger("Normalizer")

MIN_IMAGE_SIDE_PT = 40.0
EMU_PER_POINT = 12700
DECORATIVE_IMAGES_FILE = Path(__file__).with_name("decorative_images.json")


def file_id_for(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"doc_{digest[:12]}"


def image_hash(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


@cache
def decorative_hashes() -> frozenset[str]:
    if not DECORATIVE_IMAGES_FILE.is_file():
        return frozenset()
    data = json.loads(DECORATIVE_IMAGES_FILE.read_text(encoding="utf-8"))
    return frozenset(entry["sha256"] for entry in data.get("images", []))


def save_chunks(chunks: list[ExtractionInput], path: str | Path) -> Path:
    path = Path(path)
    data = [c.model_dump(mode="json") for c in chunks]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_image(blob: bytes) -> Image.Image | None:
    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        logger.info("Could not open embedded image: %s", exc)
        return None

    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, "white")
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img.convert("RGB")


class PageTextBuilder:
    def __init__(self, page_number: int, low_conf_threshold: float = 0.0):
        self.page_number = page_number
        self.low_conf_threshold = low_conf_threshold
        self.parts: list[str] = []
        self.length = 0
        self.boxes: list[BoundingBox] = []
        self.line_open = False
        self.has_native = False
        self.ocr_confidences: list[float] = []
        self.images_ocr = 0
        self.images_skipped = 0
        self.images_decorative = 0
        self.hidden: list[HiddenContent] = []
        self.unread: list[UnreadRegion] = []
        self.errors: list[str] = []

    def _append(self, s: str):
        self.parts.append(s)
        self.length += len(s)

    def _start_word(self, text: str) -> int:
        if self.line_open:
            self._append(" ")
        start = self.length
        self._append(text)
        self.line_open = True
        return start

    def add_text(self, text: str):
        for word in text.split():
            self._start_word(word)
            self.has_native = True

    def add_hidden(self, kind: str, text: str, location: str | None = None):
        text = " ".join(text.split())
        if text:
            self.hidden.append(HiddenContent(kind=kind, text=text, location=location))

    def add_unread(self, x, y, w, h, *, coord_space="page_pt", image_ref=None, reason="unread_ink",
                   image_box=None):
        self.unread.append(
            UnreadRegion(
                page=self.page_number, x=x, y=y, w=w, h=h,
                coord_space=coord_space, image_ref=image_ref, image_box=image_box, reason=reason,
            )
        )

    def add_word(
        self,
        text: str,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        source: str = "native",
        confidence: float | None = None,
        coord_space: str = "page_pt",
        image_ref: str | None = None,
        image_box: tuple[float, float, float, float] | None = None,
    ):
        start = self._start_word(text)
        if source == "ocr" and confidence is not None:
            if sum(ch.isalnum() for ch in text) >= 2:
                self.ocr_confidences.append(confidence)
        else:
            self.has_native = True
        self.boxes.append(
            BoundingBox(
                page=self.page_number,
                text=text,
                x=x,
                y=y,
                w=w,
                h=h,
                char_start=start,
                char_end=self.length,
                source=source,
                confidence=confidence,
                coord_space=coord_space,
                image_ref=image_ref,
                image_box=image_box,
            )
        )

    def add_ocr_words(
        self,
        words: list[OcrWord],
        place: Callable[[OcrWord], tuple[float, float, float, float]],
        *,
        coord_space: str = "page_pt",
        image_ref: str | None = None,
    ):
        current_block = None
        current_line = None
        for word in sorted(words, key=lambda w: (w.block, w.par, w.line, w.left)):
            if current_block is not None and word.block != current_block:
                self.end_block()
            elif (word.block, word.par, word.line) != current_line:
                self.end_line()
            current_block, current_line = word.block, (word.block, word.par, word.line)
            x, y, w, h = place(word)
            self.add_word(
                word.text, x, y, w, h,
                source="ocr", confidence=word.conf,
                coord_space=coord_space, image_ref=image_ref,
                image_box=(word.left, word.top, word.width, word.height) if image_ref else None,
            )
        self.end_block()

    def end_line(self):
        if self.line_open:
            self._append("\n")
            self.line_open = False

    def end_block(self):
        self.end_line()
        if self.parts and not "".join(self.parts[-2:]).endswith("\n\n"):
            self._append("\n")

    @property
    def is_empty(self) -> bool:
        return self.length == 0

    @property
    def text(self) -> str:
        return "".join(self.parts).rstrip("\n")

    def page_data(self) -> dict:
        has_ocr = bool(self.ocr_confidences)
        if has_ocr and self.has_native:
            method = "hybrid"
        elif has_ocr or (self.images_ocr and not self.has_native):
            method = "ocr"
        else:
            method = "native"

        confs = self.ocr_confidences
        return {
            "raw_text": self.text,
            "bounding_boxes": self.boxes,
            "ocr_used": self.images_ocr > 0,
            "extraction_method": method,
            "ocr_mean_confidence": sum(confs) / len(confs) if confs else None,
            "low_confidence_word_count": sum(c < self.low_conf_threshold for c in confs),
            "images_ocr_count": self.images_ocr,
            "images_skipped_count": self.images_skipped,
            "images_decorative_count": self.images_decorative,
            "hidden_content": self.hidden,
            "unread_regions": self.unread,
            "extraction_errors": self.errors,
        }


@dataclass
class ImageOcrOutcome:
    status: str
    result: OcrResult | None = None
    size: tuple[int, int] | None = None
    error: str | None = None


class ImageOcrCache:
    def __init__(self, ocr_engine: OcrEngine):
        self.ocr_engine = ocr_engine
        self.available = ocr_engine.is_available()
        self._cache: dict[str, ImageOcrOutcome] = {}

    def prefetch(self, items: list[tuple[str, bytes, float | None]], workers: int = OCR_WORKERS):
        todo = {ref: (blob, width) for ref, blob, width in items if ref not in self._cache}
        if not todo or not self.available:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {ref: pool.submit(self._compute, blob, width) for ref, (blob, width) in todo.items()}
        for ref, future in futures.items():
            self._cache[ref] = future.result()

    def ocr(self, ref: str, blob: bytes, display_width_pt: float | None = None) -> ImageOcrOutcome:
        if ref not in self._cache:
            self._cache[ref] = self._compute(blob, display_width_pt)
        return self._cache[ref]

    def _compute(self, blob: bytes, display_width_pt: float | None) -> ImageOcrOutcome:
        if image_hash(blob) in decorative_hashes():
            outcome = ImageOcrOutcome("decorative")
        elif not self.available:
            outcome = ImageOcrOutcome("no_ocr")
        else:
            img = load_image(blob)
            if img is None:
                outcome = ImageOcrOutcome("unreadable")
            else:
                ppi = img.width / (display_width_pt / 72) if display_width_pt else None
                try:
                    outcome = ImageOcrOutcome(
                        "ok", self.ocr_engine.ocr_image_detailed(img, px_per_inch=ppi), img.size
                    )
                except RuntimeError as exc:
                    outcome = ImageOcrOutcome("error", error=str(exc))
        return outcome


def record_image_outcome(
    builder: PageTextBuilder,
    outcome: ImageOcrOutcome,
    ref: str,
    place: Callable[[OcrWord], tuple[float, float, float, float]],
    place_box: Callable[[tuple[float, float, float, float]], tuple[float, float, float, float]],
    *,
    coord_space: str,
    whole_image: tuple[float, float, float, float] | None = None,
):
    if outcome.status == "decorative":
        builder.images_decorative += 1
        return
    if outcome.status != "ok":
        builder.images_skipped += 1
        if outcome.status == "error":
            builder.errors.append(f"OCR failed for {ref}: {outcome.error}")
        if outcome.status in ("unreadable", "error") and whole_image is not None:
            builder.add_unread(*whole_image, coord_space=coord_space, image_ref=ref,
                               reason=f"image_{outcome.status}")
        return

    builder.images_ocr += 1
    builder.add_ocr_words(outcome.result.words, place, coord_space=coord_space, image_ref=ref)
    for region in outcome.result.unread_regions:
        x0, y0, x1, y1 = region
        builder.add_unread(*place_box(region), coord_space=coord_space, image_ref=ref,
                           image_box=(x0, y0, x1 - x0, y1 - y0))


_CORE_FIELDS = {
    "creator": "author", "lastModifiedBy": "last_modified_by", "title": "title",
    "subject": "subject", "keywords": "keywords", "description": "comments",
    "category": "category", "identifier": "identifier", "contentStatus": "content_status",
}
_APP_FIELDS = {"Company": "company", "Manager": "manager", "HyperlinkBase": "hyperlink_base"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def ooxml_hidden_content(path: Path) -> list[HiddenContent]:
    items: list[HiddenContent] = []
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()

        def xml(name):
            try:
                return ElementTree.fromstring(zf.read(name))
            except (KeyError, ElementTree.ParseError):
                return None

        core = xml("docProps/core.xml")
        if core is not None:
            for el in core:
                field = _CORE_FIELDS.get(_local(el.tag))
                if field and (el.text or "").strip():
                    items.append(HiddenContent(kind="metadata", text=el.text.strip(), location=field))

        app = xml("docProps/app.xml")
        if app is not None:
            for el in app:
                field = _APP_FIELDS.get(_local(el.tag))
                if field and (el.text or "").strip():
                    items.append(HiddenContent(kind="metadata", text=el.text.strip(), location=field))

        custom = xml("docProps/custom.xml")
        if custom is not None:
            for prop in custom:
                values = [v.text.strip() for v in prop.iter() if v is not prop and (v.text or "").strip()]
                if values:
                    items.append(HiddenContent(
                        kind="metadata", text=" ".join(values),
                        location=f"custom:{prop.get('name', '')}",
                    ))

        for name in names:
            lowered = name.lower()
            if "/embeddings/" in lowered:
                items.append(HiddenContent(
                    kind="embedded_object", text=name.rsplit("/", 1)[-1], location=name,
                ))
            elif name.startswith("[trash]/"):
                text = _printable_strings(zf.read(name))
                if text:
                    items.append(HiddenContent(kind="orphan_part", text=text, location=name))
    return items


def _printable_strings(blob: bytes, min_length: int = 4) -> str:
    found = re.findall(rb"[\x20-\x7e]{%d,}" % min_length, blob)
    found += [m.decode("utf-16-le", "ignore").encode() for m in re.findall(
        rb"(?:[\x20-\x7e]\x00){%d,}" % min_length, blob)]
    return " ".join(dict.fromkeys(s.decode("ascii", "ignore") for s in found))


def drawing_part_text(blob: bytes) -> str:
    try:
        root = ElementTree.fromstring(blob)
    except ElementTree.ParseError:
        return ""
    texts = []
    for el in root.iter():
        name = _local(el.tag)
        if name == "t" and el.text:
            texts.append(el.text)
        elif name == "strCache":
            texts.extend(v.text for v in el.iter() if _local(v.tag) == "v" and v.text)
    return " ".join(dict.fromkeys(t.strip() for t in texts if t.strip()))


def extract_document(path: str | Path, **kwargs) -> list[ExtractionInput]:
    from extraction.validation import validate_document

    kind = validate_document(path)
    if kind == "pdf":
        from extraction.pdf_extractor import extract_pdf
        return extract_pdf(path, **kwargs)
    if kind == "docx":
        from extraction.docx_extractor import extract_docx
        return extract_docx(path, **kwargs)
    from extraction.pptx_extractor import extract_pptx
    return extract_pptx(path, **kwargs)


__all__ = [
    "EMU_PER_POINT",
    "MIN_IMAGE_SIDE_PT",
    "ImageOcrCache",
    "ImageOcrOutcome",
    "PageTextBuilder",
    "decorative_hashes",
    "drawing_part_text",
    "extract_document",
    "file_id_for",
    "image_hash",
    "is_meaningful",
    "load_image",
    "ooxml_hidden_content",
    "record_image_outcome",
    "save_chunks",
]
