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
    closest_anchor: str | None = None
    classify_method: str
    similarity: float
    input_text: str
    
class ClassifyBatchResponse(BaseModel):
    results: List[ClassifyResponse]

class CompensationResponse(BaseModel):
    min_salary: float | None = None
    max_salary: float | None = None
    currency: str | None = None
    currency_symbol: str | None = None
    offers_equity: bool = False
    text: str | None = None