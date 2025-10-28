import os
import logging
from urllib import response
from fastapi import FastAPI
import uvicorn # type: ignore
from schemas import Item, Items, HealthResponse, EmbedResponse, ClassifyResponse, ClassifyBatchResponse, CompensationResponse
from embedding_service import embedding_service
from compensation_parser import CompensationParser

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.DEBUG)

# ---------------------- FastAPI ----------------------
app = FastAPI()

# ---------------------- Config ----------------------
THRESHOLD_DEFAULT = float(os.getenv("SIMILARITY_THRESHOLD", 0.60))

# ---------------------- Endpoints ----------------------
@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint for monitoring"""
    return HealthResponse(status="healthy", service="embedding-service")

@app.post("/embed", response_model=EmbedResponse)
def embed(item: Item) -> EmbedResponse:
    vec = embedding_service.embed_or_cache(item.text)
    return EmbedResponse(embedding=vec.tolist())

@app.post("/classify", response_model=ClassifyResponse)
def classify(item: Item, threshold: float = THRESHOLD_DEFAULT) -> ClassifyResponse:
    logging.info(f"Classifying item with text '{item.text}' and threshold {threshold}")
    return embedding_service.classify(item.text, threshold)

@app.post("/classify_batch", response_model=ClassifyBatchResponse)
def classify_batch(items: Items, threshold: float = THRESHOLD_DEFAULT) -> ClassifyBatchResponse:
    logging.info(f"Classifying batch of {len(items.texts)} items with threshold {threshold}")
    results = embedding_service.classify_batch(items.texts, threshold)
    return ClassifyBatchResponse(results=results)

@app.post("/parse_compensation", response_model=CompensationResponse)
def parse_compensation(item: Item) -> CompensationResponse:
    logging.info(f"Parsing compensation from item with text '{item.text}'")
    parser = CompensationParser(model_name="microsoft/Phi-3-mini-4k-instruct")
    return parser.parse(item.text)

@app.post("/parse_compensation_batch", response_model=list[CompensationResponse])
def parse_compensation_batch(items: Items) -> list[CompensationResponse]:
    logging.info(f"Parsing compensation from batch of {len(items.texts)} items")
    parser = CompensationParser(model_name="microsoft/Phi-3-mini-4k-instruct")
    return parser.parse_batch(items.texts)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
