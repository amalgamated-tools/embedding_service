import os
import logging
from fastapi import FastAPI
import numpy as np
import uvicorn
from schemas import Item, Items
from embedding_service import embedding_service

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO)

# ---------------------- FastAPI ----------------------
app = FastAPI()

# ---------------------- Config ----------------------
THRESHOLD_DEFAULT = float(os.getenv("SIMILARITY_THRESHOLD", 0.70))

# ---------------------- Endpoints ----------------------
@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "embedding-service"}


@app.post("/embed")
def embed(item: Item):
    vec = embedding_service.embed_or_cache(item.text)
    return {"embedding": vec.tolist()}

@app.post("/classify")
def classify(item: Item, threshold: float = THRESHOLD_DEFAULT):
    vec = embedding_service.embed_or_cache(item.text)
    sims = np.dot(embedding_service.anchor_vecs, vec)
    idx = int(np.argmax(sims))
    category = embedding_service.anchors[idx] if sims[idx] >= threshold else "Unsure"
    # Map all engineering variants back to "Engineering"
    if category in ["Product Engineering", "Platform Engineering"]:
        category = "Engineering"
    return {"category": category, "similarity": float(sims[idx])}

@app.post("/classify_batch")
def classify_batch(items: Items, threshold: float = THRESHOLD_DEFAULT):
    results = []
    for text in items.texts:
        vec = embedding_service.embed_or_cache(text)
        sims = np.dot(embedding_service.anchor_vecs, vec)
        idx = int(np.argmax(sims))
        category = embedding_service.anchors[idx] if sims[idx] >= threshold else "Unsure"
        if category in ["Product Engineering", "Platform Engineering"]:
            category = "Engineering"
        results.append(
            {"text": text, "category": category, "similarity": float(sims[idx])}
        )
    return {"results": results}

# ---------------------- Main ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
