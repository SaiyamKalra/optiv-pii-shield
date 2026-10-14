from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    page: int
    text: str
    x: float
    y: float
    w: float
    h: float
    char_start: int | None = Field(
        None, description="Start offset of this word in the page raw_text"
    )
    char_end: int | None = Field(
        None, description="End offset (exclusive) of this word in the page raw_text"
    )
    source: str = Field(
        "native", description="'native' text layer or 'ocr'"
    )
    confidence: float | None = Field(
        None, description="Tesseract word confidence 0-100; None for native text"
    )
    coord_space: str = Field(
        "page_pt", description="'page_pt' (page/slide points) or 'image_px' (pixels in image_ref)"
    )
    image_ref: str | None = Field(
        None, description="Embedded image the word was OCR'd from, e.g. '/word/media/image3.png'"
    )
    image_box: tuple[float, float, float, float] | None = Field(
        None, description="(x, y, w, h) in pixels inside image_ref, whatever coord_space is"
    )


class HiddenContent(BaseModel):
    kind: str = Field(
        ..., description="e.g. metadata, comment, footnote, endnote, alt_text, "
                         "deleted_text, annotation, form_field, bookmark, embedded_file"
    )
    text: str
    location: str | None = Field(None, description="Where in the file, e.g. 'author' or 'comment 3'")


class UnreadRegion(BaseModel):
    page: int
    x: float
    y: float
    w: float
    h: float
    coord_space: str = "page_pt"
    image_ref: str | None = None
    image_box: tuple[float, float, float, float] | None = None
    reason: str = "unread_ink"


class ExtractionInput(BaseModel):
    file_id: str
    file_name: str = Field(
        ..., description="Name of the uploaded file"
    )
    source_url: str | None = Field(
        None, description="URL or storage path of the original file"
    )
    user_id: str | None = Field(
        None, description="ID or email of the uploading user"
    )
    page_number:int = Field(
        default=1, description="1-based page or slide index"
    )
    raw_text: str
    bounding_boxes: list[BoundingBox] | None = None
    ocr_used: bool = False
    extraction_method: str = Field(
        "native", description="'native', 'ocr' or 'hybrid'"
    )
    ocr_mean_confidence: float | None = None
    low_confidence_word_count: int = 0
    images_ocr_count: int = 0
    images_skipped_count: int = 0
    images_decorative_count: int = Field(
        0, description="Images on the decorative allowlist (logos), not OCR'd"
    )
    hidden_content: list[HiddenContent] = Field(default_factory=list)
    unread_regions: list[UnreadRegion] = Field(default_factory=list)
    extraction_errors: list[str] = Field(
        default_factory=list, description="Problems extracting this page; treat as not inspected"
    )


class DetectedEntityDetail(BaseModel):
    token: str
    entity_type: str
    original_value: str
    start: int
    end: int
    score: float
    page_number: int=1

class PIIOutput(BaseModel):
    file_id: str
    file_name: str
    source_url: str | None = None
    user_id: str | None = None
    page_number: int = 1
    status: str
    sanitized_text: str
    mapping_ledger: dict[str, str] = Field(
        default_factory=dict,
        description="Token -> Original Value mapping"
    )
    detected_entities: list[DetectedEntityDetail] = Field(
        default_factory=list
    )
    verification_passed: bool
    detection_counts: dict[str, int] = Field(default_factory=dict)