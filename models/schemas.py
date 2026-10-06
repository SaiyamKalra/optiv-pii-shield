from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """One word on a page. x, y, w, h are PDF points with a top-left origin."""

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