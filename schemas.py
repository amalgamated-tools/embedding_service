from pydantic import BaseModel
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
    text: str
    threshold: float
    category: str
    closest_anchor: str
    category_before_mapping: str
    similarity: float


class ClassifyBatchItem(BaseModel):
    text: str
    threshold: float
    category: str
    closest_anchor: str
    category_before_mapping: str
    similarity: float


class ClassifyBatchResponse(BaseModel):
    results: List[ClassifyBatchItem]