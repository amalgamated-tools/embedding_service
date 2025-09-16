import os
import re
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
import redis
import json
from functools import lru_cache
import uvicorn

# ---------------------- Logging ----------------------
logging.basicConfig(level=logging.INFO)

# ---------------------- FastAPI ----------------------
app = FastAPI()

# ---------------------- Config ----------------------
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
THRESHOLD_DEFAULT = float(os.getenv("SIMILARITY_THRESHOLD", 0.70))

# ---------------------- Load model ----------------------
logging.info(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# ---------------------- Anchors ----------------------
anchors = [
    "Engineering",
    "Product Engineering",
    "Platform Engineering",
    "Non-Engineering"
]
anchor_vecs = model.encode(anchors, normalize_embeddings=True)

# ---------------------- Redis / Fallback ----------------------
use_redis = True
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
    r.ping()
    logging.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except redis.RedisError:
    logging.warning("Redis unavailable; falling back to in-memory LRU cache")
    use_redis = False

# ---------------------- In-memory cache ----------------------
@lru_cache(maxsize=5000)
def _embed(text: str):
    return model.encode([text], normalize_embeddings=True)[0]

def cached_embed_lru(text: str):
    info_before = _embed.cache_info()
    vec = _embed(text)
    info_after = _embed.cache_info()
    if info_after.hits > info_before.hits:
        logging.info(f"Cache HIT (LRU): {text}")
    else:
        logging.info(f"Cache MISS (LRU): {text}")
    return vec

# ---------------------- Text normalization ----------------------
def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'[^a-z ]+', '', text)  # remove punctuation
    return text.strip()

# ---------------------- Helper: embedding w/ cache ----------------------
def embed_or_cache(text: str):
    text_norm = normalize(text)
    if use_redis:
        key = f"embed:{text_norm}"
        val = r.get(key)
        if val is not None:
            logging.info(f"Cache HIT (Redis): {text_norm}")
            return np.array(json.loads(val), dtype=np.float32)
        logging.info(f"Cache MISS (Redis): {text_norm}")
        vec = model.encode([text_norm], normalize_embeddings=True)[0]
        r.set(key, json.dumps(vec.tolist()))
        return vec
    else:
        return cached_embed_lru(text_norm)

# ---------------------- Pydantic Schemas ----------------------
class Item(BaseModel):
    text: str

class Items(BaseModel):
    texts: List[str]

# ---------------------- Endpoints ----------------------
@app.get("/health")
def health_check():
    """Health check endpoint for monitoring"""
    return {"status": "healthy", "service": "embedding-service"}


@app.post("/embed")
def embed(item: Item):
    vec = embed_or_cache(item.text)
    return {"embedding": vec.tolist()}

@app.post("/classify")
def classify(item: Item, threshold: float = THRESHOLD_DEFAULT):
    vec = embed_or_cache(item.text)
    sims = np.dot(anchor_vecs, vec)
    idx = int(np.argmax(sims))
    category = anchors[idx] if sims[idx] >= threshold else "Unsure"
    # Map all engineering variants back to "Engineering"
    if category in ["Product Engineering", "Platform Engineering"]:
        category = "Engineering"
    return {"category": category, "similarity": float(sims[idx])}

@app.post("/classify_batch")
def classify_batch(items: Items, threshold: float = THRESHOLD_DEFAULT):
    results = []
    for text in items.texts:
        vec = embed_or_cache(text)
        sims = np.dot(anchor_vecs, vec)
        idx = int(np.argmax(sims))
        category = anchors[idx] if sims[idx] >= threshold else "Unsure"
        if category in ["Product Engineering", "Platform Engineering"]:
            category = "Engineering"
        results.append(
            {"text": text, "category": category, "similarity": float(sims[idx])}
        )
    return {"results": results}

# ---------------------- Main ----------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
