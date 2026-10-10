# ocr/ocr_engine.py

import logging
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pytesseract
from dotenv import load_dotenv
from PIL import Image, ImageOps, ImageStat
from pytesseract import Output

logger = logging.getLogger("OcrEngine")

DEFAULT_WINDOWS_TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Images shorter than this (in pixels) are upscaled before OCR so small text is legible.
MIN_OCR_HEIGHT_PX = 1000
UPSCALE_FACTOR = 2
# Mean grayscale below this means a dark background with light text -> invert.
DARK_IMAGE_THRESHOLD = 110
LOW_CONF_THRESHOLD = 60.0
# Characters Tesseract may not output. "|" is almost never real text in a scan, but
# Tesseract picks it for a lowercase "l", which breaks emails like "l.brannstrom@...".
CHAR_BLACKLIST = "|"
# Share of the CPUs used for OCR when OCR_WORKERS isn't set (leaves headroom for the rest
# of the machine), and an upper bound so memory for queued page images stays modest.
OCR_CPU_SHARE = 0.75
OCR_MAX_WORKERS = 16
# Seconds before a single Tesseract call is abandoned.
OCR_TIMEOUT_S = 180

# Second pass on lines that contain low-confidence words.
LINE_RETRY_SCALE = 3.0
LINE_RETRY_MIN_GAIN = 5.0

# Finding ink (dark marks) that no OCR word covers.
INK_LEVEL = 128               # grayscale below this counts as ink
INK_TILE_PX = 24              # analysis grid cell size
INK_TILE_MIN_PIXELS = 12      # a cell needs at least this much leftover ink
INK_REGION_MIN_TILES = 3      # smaller leftovers are specks, not content
INK_MAX_REGIONS = 15          # bound the extra OCR work per image
INK_MIN_SIDE_PX = 6           # thinner leftovers are rule fragments
INK_SOLID_FILL = 0.85         # ink density above this is a filled band, not marks
DARK_BAND = 0.6               # regions this dark hold light text on a dark fill
LINE_RETRY_MAX = 30           # most low-confidence lines re-read per image
ORIENTATION_MIN_GAIN = 1.5    # a rotation must read this much better to be used
LONG_LINE_FRACTION = 0.08     # straight runs longer than this share of the image are rules/borders


def _usable_cpus() -> int:
    """CPUs this process may run on (respects affinity/container limits where Python can see them)."""
    counter = getattr(os, "process_cpu_count", None)  # Python 3.13+
    if counter is not None:
        return counter() or 1
    if hasattr(os, "sched_getaffinity"):  # Linux
        return len(os.sched_getaffinity(0)) or 1
    return os.cpu_count() or 1


def default_ocr_workers() -> int:
    """How many images to OCR at once on this machine.

    OCR_WORKERS (environment or .env) wins when it is a positive integer; otherwise
    75% of the usable CPUs, at least 1 and at most OCR_MAX_WORKERS.
    """
    load_dotenv()
    configured = os.environ.get("OCR_WORKERS", "").strip()
    if configured:
        try:
            value = int(configured)
            if value > 0:
                return value
        except ValueError:
            pass
        logger.warning("Ignoring invalid OCR_WORKERS=%r", configured)
    return max(1, min(OCR_MAX_WORKERS, round(_usable_cpus() * OCR_CPU_SHARE)))


def resolve_tesseract_cmd() -> str | None:
    """Find the Tesseract binary: TESSERACT_CMD (.env or environment), PATH, default install."""
    load_dotenv()
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd and os.path.exists(env_cmd):
        return env_cmd

    on_path = shutil.which("tesseract")
    if on_path:
        return on_path

    if os.path.exists(DEFAULT_WINDOWS_TESSERACT):
        return DEFAULT_WINDOWS_TESSERACT
    return None


@dataclass
class OcrWord:
    """A word recognised by Tesseract. Coordinates are pixels in the input image."""

    text: str
    left: float
    top: float
    width: float
    height: float
    conf: float
    block: int
    par: int
    line: int


@dataclass
class OcrResult:
    """Words read from an image, plus areas holding ink that could not be read.

    unread_regions are (x0, y0, x1, y1) pixel boxes in the input image.
    """

    words: list[OcrWord]
    unread_regions: list[tuple[float, float, float, float]] = field(default_factory=list)


def is_meaningful(word: OcrWord) -> bool:
    """Words with no letters or digits (table rules, bullets) can't hold PII."""
    return any(ch.isalnum() for ch in word.text)


class OcrEngine:

    def __init__(
        self,
        tesseract_cmd: str | None = None,
        lang: str = "eng",
        psm: int = 3,
        low_conf_threshold: float = LOW_CONF_THRESHOLD,
        timeout: float = OCR_TIMEOUT_S,
    ):
        self.tesseract_cmd = tesseract_cmd or resolve_tesseract_cmd()
        self.lang = lang
        self.psm = psm
        self.low_conf_threshold = low_conf_threshold
        self.timeout = timeout
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check this engine's own binary (pytesseract caches its version check globally)."""
        if self._available is None:
            self._available = False
            cmd = self.tesseract_cmd and (shutil.which(self.tesseract_cmd) or self.tesseract_cmd)
            if cmd and os.path.isfile(cmd):
                try:
                    subprocess.run(
                        [cmd, "--version"], capture_output=True, timeout=30, check=True
                    )
                    self._available = True
                except (OSError, subprocess.SubprocessError):
                    pass
        return self._available

    def preprocess(self, img: Image.Image, scale: float = 1.0) -> tuple[Image.Image, float]:
        """Grayscale, invert dark images, stretch contrast, upscale small ones.

        Returns the processed image and the total scale factor applied, so word
        boxes can be mapped back to the original image's pixels.
        """
        gray = img.convert("L")

        if ImageStat.Stat(gray).mean[0] < DARK_IMAGE_THRESHOLD:
            gray = ImageOps.invert(gray)
        # Stretch contrast so faint grey text reads as dark text.
        gray = ImageOps.autocontrast(gray, cutoff=1)

        if gray.height < MIN_OCR_HEIGHT_PX:
            scale *= UPSCALE_FACTOR

        if scale != 1.0:
            gray = gray.resize(
                (round(gray.width * scale), round(gray.height * scale)),
                Image.Resampling.LANCZOS,
            )
        return gray, scale

    def _run(self, img: Image.Image, scale: float, psm: int | None = None) -> list[OcrWord]:
        processed, scale = self.preprocess(img, scale)
        return self._tesseract(processed, scale, psm)

    def _tesseract(self, processed: Image.Image, scale: float, psm: int | None = None) -> list[OcrWord]:
        """Run Tesseract on an already preprocessed image; boxes are divided by scale."""
        # pytesseract reads the binary path from a module global; set it per call so
        # engines configured with different binaries don't interfere.
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        data = pytesseract.image_to_data(
            processed,
            lang=self.lang,
            config=f"--psm {psm or self.psm} -c tessedit_char_blacklist={CHAR_BLACKLIST}",
            output_type=Output.DICT,
            timeout=self.timeout,
        )

        words = []
        for i, text in enumerate(data["text"]):
            conf = float(data["conf"][i])
            # level 5 = word; conf -1 marks non-word layout rows
            if data["level"][i] != 5 or conf < 0 or not text.strip():
                continue
            words.append(
                OcrWord(
                    text=text.strip(),
                    left=data["left"][i] / scale,
                    top=data["top"][i] / scale,
                    width=data["width"][i] / scale,
                    height=data["height"][i] / scale,
                    conf=conf,
                    block=data["block_num"][i],
                    par=data["par_num"][i],
                    line=data["line_num"][i],
                )
            )
        return words

    def ocr_image(self, img: Image.Image) -> list[OcrWord]:
        """OCR an image; retry once at a larger scale if mean confidence is low."""
        words = self._run(img, scale=1.0)
        if mean_confidence(words) >= self.low_conf_threshold:
            return words

        retry = self._run(img, scale=float(UPSCALE_FACTOR))
        if mean_confidence(retry) > mean_confidence(words):
            logger.info(
                "Low-confidence OCR retried at %sx scale (%.1f -> %.1f)",
                UPSCALE_FACTOR, mean_confidence(words), mean_confidence(retry),
            )
            return retry
        return words

    def ocr_image_detailed(self, img: Image.Image, px_per_inch: float | None = None) -> OcrResult:
        """Full OCR: whole image (turned upright if it is stored sideways or upside
        down), second pass on hard lines and unread ink, then report whatever ink is
        still not covered by a word.

        px_per_inch is the image's real resolution where known; it lets rules and
        borders (an inch or longer) be told apart from short marks like barcode bars.
        Returned coordinates are pixels in the image as given.
        """
        angle, upright, words = self._best_orientation(img)
        words = self._refine_low_confidence_lines(upright, words)
        words = self._read_uncovered_ink(upright, words, px_per_inch)
        unread = _uncovered_ink_regions(upright, words, px_per_inch)
        if angle:
            words = [_unrotate_word(w, angle, img.size) for w in words]
            unread = [_unrotate_box(b, angle, img.size) for b in unread]
        return OcrResult(words, unread)

    def _best_orientation(self, img: Image.Image) -> tuple[int, Image.Image, list[OcrWord]]:
        """Angle (counter-clockwise degrees) that turns the text upright, the upright
        image, and its words. Only tried when the image as given reads poorly."""
        words = self.ocr_image(img)
        if mean_confidence(words) >= self.low_conf_threshold:
            return 0, img, words
        best = (_reading_score(words), 0, img, words)
        for angle in (180, 90, 270):
            turned = img.rotate(angle, expand=True)
            try:
                turned_words = self._run(turned, scale=1.0)
            except RuntimeError:
                continue
            score = _reading_score(turned_words)
            if score > best[0] * ORIENTATION_MIN_GAIN:
                best = (score, angle, turned, turned_words)
        if best[1]:
            logger.info("Image text read upright after rotating %d degrees", best[1])
        return best[1], best[2], best[3]

    # --- second pass: lines with low-confidence words ---------------------------------
    def _refine_low_confidence_lines(self, img: Image.Image, words: list[OcrWord]) -> list[OcrWord]:
        lines: dict[tuple[int, int, int], list[OcrWord]] = {}
        for w in words:
            lines.setdefault((w.block, w.par, w.line), []).append(w)

        # Only lines with a real low-confidence word (3+ characters) are worth re-reading.
        hard = [
            key for key, line_words in lines.items()
            if any(
                is_meaningful(w) and len(w.text) >= 3 and w.conf < self.low_conf_threshold
                for w in line_words
            )
        ][:LINE_RETRY_MAX]
        boxes = [_line_box(lines[key], img.size) for key in hard]
        candidates = [(key, box) for key, box in zip(hard, boxes, strict=True) if box is not None]
        rereads = self._read_crops(img, [box for _, box in candidates], LINE_RETRY_SCALE, psm=7)

        replaced = {}
        for i, (key, _) in enumerate(candidates):
            reread = rereads.get(i)
            if not reread:
                continue
            if mean_confidence(reread) < mean_confidence(lines[key]) + LINE_RETRY_MIN_GAIN:
                continue
            block, par, line = key
            for w in reread:
                w.block, w.par, w.line = block, par, line
            replaced[key] = reread

        result = []
        for key, line_words in lines.items():
            result.extend(replaced.get(key, line_words))
        return result

    # --- re-reading many small crops ---------------------------------------------------
    def _read_crops(
        self, img: Image.Image, boxes: list[tuple[int, int, int, int]], scale: float, psm: int
    ) -> dict[int, list[OcrWord]]:
        """OCR several crops of img in parallel (each Tesseract call is its own process).

        Returns words per crop index, in the original image's pixel coordinates. A failed
        crop is left out of the result, so callers keep what they had.
        """
        def read(box):
            try:
                words = self._run(img.crop(box), scale=scale, psm=psm)
            except RuntimeError as exc:
                logger.info("Re-read of crop %s failed: %s", box, exc)
                return None
            for w in words:
                w.left += box[0]
                w.top += box[1]
            return words

        results = _crop_pool().map(read, boxes) if len(boxes) > 1 else map(read, boxes)
        return {i: words for i, words in enumerate(results) if words is not None}

    # --- second pass: ink that no word covers -----------------------------------------
    def _read_uncovered_ink(self, img: Image.Image, words: list[OcrWord],
                            rule_px: float | None = None) -> list[OcrWord]:
        regions = _uncovered_ink_regions(img, words, rule_px)[:INK_MAX_REGIONS]
        if not regions:
            return words

        pad = 8
        boxes = [
            box for box in (
                _clamp_box((r[0] - pad, r[1] - pad, r[2] + pad, r[3] + pad), img.size)
                for r in regions
            )
            if box is not None
        ]
        found_by_region = self._read_crops(img, boxes, float(UPSCALE_FACTOR), psm=11)

        next_block = max((w.block for w in words), default=0) + 1
        added = []
        for index in range(len(boxes)):
            for w in found_by_region.get(index, []):
                # Only keep confident recoveries; anything else stays reported as unread.
                if w.conf < self.low_conf_threshold or not is_meaningful(w):
                    continue
                if any(_overlap(w, existing) > 0.3 for existing in words + added):
                    continue
                w.block, w.par = next_block, 1
                added.append(w)
            next_block += 1
        if added:
            logger.info("Recovered %d word(s) from unread ink", len(added))
        return words + added


def mean_confidence(words: list[OcrWord]) -> float:
    """Average confidence of words that contain letters or digits."""
    scored = [w for w in words if is_meaningful(w)] or words
    if not scored:
        return 0.0
    return sum(w.conf for w in scored) / len(scored)


def _clamp_box(box, size) -> tuple[int, int, int, int] | None:
    width, height = size
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(width, int(box[2] + 1)), min(height, int(box[3] + 1))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def _overlap(a: OcrWord, b: OcrWord) -> float:
    """Intersection area as a fraction of the smaller word."""
    ix = max(0.0, min(a.left + a.width, b.left + b.width) - max(a.left, b.left))
    iy = max(0.0, min(a.top + a.height, b.top + b.height) - max(a.top, b.top))
    smaller = min(a.width * a.height, b.width * b.height) or 1.0
    return ix * iy / smaller


def _long_runs(mask: np.ndarray, length: int) -> np.ndarray:
    """Pixels that belong to a horizontal run of at least `length` set pixels."""
    height, width = mask.shape
    if width < length:
        return np.zeros_like(mask)
    csum = np.zeros((height, width + 1), dtype=np.int32)
    np.cumsum(mask, axis=1, out=csum[:, 1:])
    full = (csum[:, length:] - csum[:, :-length]) == length  # window starts, (H, W-L+1)
    starts = np.zeros((height, full.shape[1] + 1), dtype=np.int32)
    np.cumsum(full, axis=1, out=starts[:, 1:])
    j = np.arange(width)
    lo = np.clip(j - length + 1, 0, full.shape[1])
    hi = np.clip(j + 1, 0, full.shape[1])
    return (starts[:, hi] - starts[:, lo]) > 0


def _ink_mask(img: Image.Image, rule_px: float | None = None) -> np.ndarray:
    gray = img.convert("L")
    if ImageStat.Stat(gray).mean[0] < DARK_IMAGE_THRESHOLD:
        gray = ImageOps.invert(gray)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    ink = np.asarray(gray) < INK_LEVEL

    # Remove long straight rules and borders (table lines, solid fills); keep text and
    # short marks such as barcode bars.
    length = int(rule_px) if rule_px else max(60, int(LONG_LINE_FRACTION * min(ink.shape)))
    lines = _long_runs(ink, length) | _long_runs(ink.T, length).T
    return ink & ~lines


def _uncovered_ink_regions(
    img: Image.Image, words: list[OcrWord], rule_px: float | None = None
) -> list[tuple[float, float, float, float]]:
    """Boxes (pixels) around clusters of ink that no OCR word covers."""
    ink = _ink_mask(img, rule_px)
    height, width = ink.shape
    covered = np.zeros_like(ink)
    for w in words:
        pad = max(4.0, 0.3 * w.height)
        x0, y0 = max(0, int(w.left - pad)), max(0, int(w.top - pad))
        x1, y1 = min(width, int(w.left + w.width + pad) + 1), min(height, int(w.top + w.height + pad) + 1)
        covered[y0:y1, x0:x1] = True
    ink &= ~covered

    t = INK_TILE_PX
    rows, cols = -(-height // t), -(-width // t)
    padded = np.zeros((rows * t, cols * t), dtype=bool)
    padded[:height, :width] = ink
    counts = padded.reshape(rows, t, cols, t).sum(axis=(1, 3))
    tiles = counts >= INK_TILE_MIN_PIXELS

    regions = []
    seen = np.zeros_like(tiles)
    for r, c in zip(*np.nonzero(tiles), strict=True):
        if seen[r, c]:
            continue
        stack, members = [(r, c)], []
        seen[r, c] = True
        while stack:
            cr, cc = stack.pop()
            members.append((cr, cc))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and tiles[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        if len(members) < INK_REGION_MIN_TILES:
            continue
        rs = [m[0] for m in members]
        cs = [m[1] for m in members]
        y0, y1 = min(rs) * t, min(height, (max(rs) + 1) * t)
        x0, x1 = min(cs) * t, min(width, (max(cs) + 1) * t)
        ys, xs = np.nonzero(ink[y0:y1, x0:x1])
        regions.append((x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1))

    gray = np.asarray(img.convert("L"))
    regions = [r for r in regions if _looks_like_content(r, ink, gray, covered)]
    regions.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return [tuple(float(v) for v in b) for b in regions]


def _looks_like_content(region, ink, gray, covered) -> bool:
    """Drop leftovers that are table fills or rule fragments rather than marks."""
    x0, y0, x1, y1 = (int(v) for v in region)
    short, long_side = sorted((x1 - x0, y1 - y0))
    if short < INK_MIN_SIDE_PX:
        return False
    if short < 4 * INK_MIN_SIDE_PX and long_side > 4 * short:
        return False  # a thin sliver: connector or rule fragment
    if ink[y0:y1, x0:x1].mean() > INK_SOLID_FILL:
        return False
    if _is_hollow_frame(ink[y0:y1, x0:x1]):
        return False
    if (gray[y0:y1, x0:x1] < 100).mean() > DARK_BAND:
        # Light text on a dark fill: only content if some light text is still unread.
        light_unread = (gray[y0:y1, x0:x1] > 170) & ~covered[y0:y1, x0:x1]
        return light_unread.mean() > 0.02
    return True


def _reading_score(words: list[OcrWord]) -> float:
    """Total confidence of real words: rewards reading more text, and reading it well."""
    return sum(w.conf for w in words if is_meaningful(w) and len(w.text) >= 2)


def _unrotate_box(box, angle: int, size: tuple[int, int]) -> tuple[float, float, float, float]:
    """Map an (x0, y0, x1, y1) box from an image rotated `angle` degrees counter-clockwise
    (PIL Image.rotate with expand=True) back to the original image of `size`."""
    width, height = size
    x0, y0, x1, y1 = box
    if angle == 180:
        return width - x1, height - y1, width - x0, height - y0
    if angle == 90:
        return width - y1, x0, width - y0, x1
    if angle == 270:
        return y0, height - x1, y1, height - x0
    return box


def _unrotate_word(word: OcrWord, angle: int, size: tuple[int, int]) -> OcrWord:
    x0, y0, x1, y1 = _unrotate_box(
        (word.left, word.top, word.left + word.width, word.top + word.height), angle, size
    )
    word.left, word.top, word.width, word.height = x0, y0, x1 - x0, y1 - y0
    return word


def _is_hollow_frame(ink: np.ndarray) -> bool:
    """A box outline (highlight rectangle, button border) with nothing unread inside."""
    height, width = ink.shape
    if min(height, width) < 4 * INK_MIN_SIDE_PX:
        return False
    inset_y, inset_x = max(3, int(0.15 * height)), max(3, int(0.15 * width))
    interior = ink[inset_y:height - inset_y, inset_x:width - inset_x]
    return interior.size > 0 and interior.mean() < 0.005


def _line_box(line_words: list[OcrWord], size) -> tuple[int, int, int, int] | None:
    """A line's bounding box with some padding, clamped to the image."""
    x0 = min(w.left for w in line_words)
    y0 = min(w.top for w in line_words)
    x1 = max(w.left + w.width for w in line_words)
    y1 = max(w.top + w.height for w in line_words)
    pad = max(4.0, 0.35 * (y1 - y0))
    return _clamp_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), size)



# Images OCR'd at once, decided per machine (see default_ocr_workers).
OCR_WORKERS = default_ocr_workers()

_CROP_POOL: ThreadPoolExecutor | None = None
_CROP_POOL_LOCK = threading.Lock()


def _crop_pool() -> ThreadPoolExecutor:
    """Shared pool for re-reading crops of one image in parallel.

    Separate from the per-page pools (a page task waiting on its own pool could
    deadlock), and shared so all images together stay within OCR_WORKERS extra calls.
    """
    global _CROP_POOL
    with _CROP_POOL_LOCK:
        if _CROP_POOL is None:
            _CROP_POOL = ThreadPoolExecutor(max_workers=OCR_WORKERS, thread_name_prefix="ocr-crop")
        return _CROP_POOL
