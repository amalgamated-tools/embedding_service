from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
import uvicorn
import logging

# ---------------------- Setup ----------------------
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# Load a higher-quality model for semantic similarity
model = SentenceTransformer("all-mpnet-base-v2")

# Define anchors
anchors = ["Engineering", "Non-Engineering"]
anchor_vecs = model.encode(anchors, normalize_embeddings=True)

# ---------------------- LRU Cache ----------------------
@lru_cache(maxsize=5000)
def _embed(text: str):
    return model.encode([text], normalize_embeddings=True)[0]

def cached_embed(text: str):
    """Return embedding, logging cache hits/misses"""
    info_before = _embed.cache_info()
    vec = _embed(text)
    info_after = _embed.cache_info()
    if info_after.hits > info_before.hits:
        logging.info(f"Cache HIT: {text}")
    else:
        logging.info(f"Cache MISS: {text}")
    return vec

# ---------------------- Pydantic Schemas ----------------------
class Item(BaseModel):
    text: str

class Items(BaseModel):
    texts: List[str]

# ---------------------- Endpoints ----------------------
@app.post("/embed")
def embed(item: Item):
    vec = cached_embed(item.text)
    return {"embedding": vec.tolist()}

@app.post("/classify")
def classify(item: Item, threshold: float = 0.75):
    vec = cached_embed(item.text)
    sims = np.dot(anchor_vecs, vec)
    idx = int(np.argmax(sims))
    category = anchors[idx] if sims[idx] >= threshold else "Unsure"
    return {"category": category, "similarity": float(sims[idx])}

@app.post("/classify_batch")
def classify_batch(items: Items, threshold: float = 0.75):
    results = []
    for text in items.texts:
        vec = cached_embed(text)
        sims = np.dot(anchor_vecs, vec)
        idx = int(np.argmax(sims))
        category = anchors[idx] if sims[idx] >= threshold else "Unsure"
        results.append(
            {"text": text, "category": category, "similarity": float(sims[idx])}
        )
    return {"results": results}

# ---------------------- Main ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
