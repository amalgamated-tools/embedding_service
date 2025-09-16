from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np
from functools import lru_cache
import uvicorn

app = FastAPI()

# Load model once at startup
model = SentenceTransformer("all-MiniLM-L6-v2")

anchors = ["Engineering", "Non-Engineering"]
anchor_vecs = model.encode(anchors, normalize_embeddings=True)

# ----------- Caching -----------
@lru_cache(maxsize=5000)
def cached_embed(text: str):
    """Return an embedding vector, cached by text."""
    return model.encode([text], normalize_embeddings=True)[0]


class Item(BaseModel):
    text: str


class Items(BaseModel):
    texts: List[str]


@app.post("/embed")
def embed(item: Item):
    vec = cached_embed(item.text)
    return {"embedding": vec.tolist()}


@app.post("/classify")
def classify(item: Item):
    vec = cached_embed(item.text)
    sims = np.dot(anchor_vecs, vec)
    idx = int(np.argmax(sims))
    return {"category": anchors[idx], "similarity": float(sims[idx])}


@app.post("/classify_batch")
def classify_batch(items: Items):
    results = []
    for text in items.texts:
        vec = cached_embed(text)
        sims = np.dot(anchor_vecs, vec)
        idx = int(np.argmax(sims))
        results.append(
            {"text": text, "category": anchors[idx], "similarity": float(sims[idx])}
        )
    return {"results": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
