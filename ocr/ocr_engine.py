# ocr/ocr_engine.py

import logging
import os
import shutil
from dataclasses import dataclass

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


class OcrEngine:

    def __init__(
        self,
        tesseract_cmd: str | None = None,
        lang: str = "eng",
        psm: int = 3,
        low_conf_threshold: float = LOW_CONF_THRESHOLD,
    ):
        self.tesseract_cmd = tesseract_cmd or resolve_tesseract_cmd()
        self.lang = lang
        self.psm = psm
        self.low_conf_threshold = low_conf_threshold
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

    def is_available(self) -> bool:
        if not self.tesseract_cmd:
            return False
        try:
            pytesseract.get_tesseract_version()
            return True
        except (pytesseract.TesseractNotFoundError, OSError):
            return False

    def preprocess(self, img: Image.Image, scale: float = 1.0) -> tuple[Image.Image, float]:
        """Grayscale, invert dark images, upscale small ones.

        Returns the processed image and the total scale factor applied, so word
        boxes can be mapped back to the original image's pixels.
        """
        gray = img.convert("L")

        if ImageStat.Stat(gray).mean[0] < DARK_IMAGE_THRESHOLD:
            gray = ImageOps.invert(gray)

        if gray.height < MIN_OCR_HEIGHT_PX:
            scale *= UPSCALE_FACTOR

        if scale != 1.0:
            gray = gray.resize(
                (round(gray.width * scale), round(gray.height * scale)),
                Image.Resampling.LANCZOS,
            )
        return gray, scale

    def _run(self, img: Image.Image, scale: float) -> list[OcrWord]:
        processed, scale = self.preprocess(img, scale)
        data = pytesseract.image_to_data(
            processed,
            lang=self.lang,
            config=f"--psm {self.psm}",
            output_type=Output.DICT,
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


def mean_confidence(words: list[OcrWord]) -> float:
    if not words:
        return 0.0
    return sum(w.conf for w in words) / len(words)
