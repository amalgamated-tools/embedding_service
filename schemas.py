from pydantic import BaseModel # type: ignore
from typing import List
class Item(BaseModel):
    text: str

class Items(BaseModel):
    texts: List[str]


# Response models
class HealthResponse(BaseModel):
    status: str
    service: str

class EmbedResponse(BaseModel):
    embedding: List[float]
    
class ClassifyResponse(BaseModel):
    category_before_mapping: str
    mapped_category: str
    # this can be None if the classify method is not embedding based
    closest_anchor: str | None = None
    classify_method: str
    similarity: float
    input_text: str

class ClassifyBatchResponse(BaseModel):
    results: List[ClassifyResponse]