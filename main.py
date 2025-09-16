import os
import logging
from venv import logger
from fastapi import FastAPI
import numpy as np
import uvicorn
from schemas import Item, Items, HealthResponse, EmbedResponse, ClassifyResponse, ClassifyBatchResponse
from embedding_service import embedding_service

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.DEBUG)

# ---------------------- FastAPI ----------------------
app = FastAPI()

# ---------------------- Config ----------------------
THRESHOLD_DEFAULT = float(os.getenv("SIMILARITY_THRESHOLD", 0.64))

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
    vec = embedding_service.embed_or_cache(item.text)
    sims = np.dot(embedding_service.anchor_vecs, vec)
    idx = int(np.argmax(sims))

    anchor = embedding_service.anchors[idx]
    unmapped_category = anchor if sims[idx] >= threshold else "Unsure"
    category = embedding_service.map_category(unmapped_category)

    return ClassifyResponse(
        text=item.text,
        threshold=threshold,
        category=category,
        closest_anchor=anchor,
        category_before_mapping=unmapped_category,
        similarity=float(sims[idx])
    )

@app.post("/classify_batch", response_model=ClassifyBatchResponse)
def classify_batch(items: Items, threshold: float = THRESHOLD_DEFAULT) -> ClassifyBatchResponse:
    logging.info(f"Classifying batch of {len(items.texts)} items with threshold {threshold}")
    results = []
    for text in items.texts:
        vec = embedding_service.embed_or_cache(text)
        sims = np.dot(embedding_service.anchor_vecs, vec)
        idx = int(np.argmax(sims))
        
        anchor = embedding_service.anchors[idx]
        unmapped_category = anchor if sims[idx] >= threshold else "Unsure"
        category = embedding_service.map_category(unmapped_category)
        
        results.append({
            "text": text,
            "threshold": threshold,
            "category": category,
            "closest_anchor": anchor,
            "category_before_mapping": unmapped_category,
            "similarity": float(sims[idx])
        })
    return ClassifyBatchResponse(results=results)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
