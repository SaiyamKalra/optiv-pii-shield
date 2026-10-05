from typing import Dict,List,Optimal
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    page:int
    text:str
    x:float
    y:float
    w:float
    h:float

class ExtractionInput(BaseModel):
    file_id:str
    file_name:str=Field(
        ...,description="Name of the uploaded file"
    )
    source_url:Optional[str]=Field(
        None, description="URL or storage path of the original file"
    )
    raw_text:str
    bounding_boxes: Optional[List[BoundingBox]] = None

class DetectedEntityDetail(BaseModel):
    token:str
    entity_type:str
    original_value:str
    start:int
    end:int
    score:float

class PIIOutput(BaseModel):
    file_id:str
    file_name:str
    source_url:Optional[str]=None
    status: str
    sanitized_text: str
    mapping_ledger: Dict[str, str] = Field(
        default_factory=dict, description="Token -> Original Value mapping"
    )
    detected_entities: List[DetectedEntityDetail] = Field(default_factory=list)
    verification_passed: bool
    detection_counts: Dict[str, int] = Field(default_factory=dict)