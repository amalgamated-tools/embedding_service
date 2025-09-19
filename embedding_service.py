import os
import re
import logging
from sentence_transformers import SentenceTransformer
import numpy as np
import redis
import json
from functools import lru_cache


class EmbeddingService:
    def __init__(self):
        # Configuration
        self.model_name = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
        self.redis_host = os.getenv("REDIS_HOST", "redis")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        
        # Load model
        logging.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Anchors
        self.anchors = [
            "Design",
            "Engineering",
            "Non-Engineering"
            "Product",
            "Sales",
        ]
        self.anchor_vecs = self.model.encode(self.anchors, normalize_embeddings=True)

        self.variant_map = {
            "Product Engineering": "Engineering",
            "Platform Engineering": "Engineering",
            "Software Engineering": "Engineering",
            "Product Manager": "Product",
            "UX Designer": "Design",
            "Sales Associate": "Sales"
        }
        
        # Redis setup
        self._setup_redis()
        
    def _setup_redis(self):
        """Setup Redis connection with fallback to in-memory cache"""
        self.use_redis = True
        try:
            self.r = redis.Redis(host=self.redis_host, port=self.redis_port, decode_responses=False)
            self.r.ping()
            logging.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except redis.RedisError:
            logging.warning("Redis unavailable; falling back to in-memory LRU cache")
            self.use_redis = False
    
    @lru_cache(maxsize=5000)
    def _embed(self, text: str):
        """Internal embedding function with LRU cache"""
        return self.model.encode([text], normalize_embeddings=True)[0]

    def cached_embed_lru(self, text: str):
        """Embedding with LRU cache and logging"""
        info_before = self._embed.cache_info()
        vec = self._embed(text)
        info_after = self._embed.cache_info()
        if info_after.hits > info_before.hits:
            logging.info(f"Cache HIT (LRU): {text}")
        else:
            logging.info(f"Cache MISS (LRU): {text}")
        return vec

    def normalize(self, text: str) -> str:
        """Normalize text for consistent embedding"""
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # remove numbers
        text = re.sub(r'[^a-z ]+', '', text)  # remove punctuation
        return text.strip()

    def map_category(self, category: str) -> str:
        logging.info(f"Mapping category: {category}")
        return self.variant_map.get(self.normalize(category), category)

    def embed_or_cache(self, text: str):
        """Get embedding with Redis or LRU cache"""
        text_norm = self.normalize(text)
        logging.info(f"Normalized text: {text_norm}")
        if self.use_redis:
            key = f"embed:{text_norm}"
            val = self.r.get(key)
            if val is not None:
                logging.info(f"Cache HIT (Redis): {text_norm}")
                return np.array(json.loads(val), dtype=np.float32)
            logging.info(f"Cache MISS (Redis): {text_norm}")
            vec = self.model.encode([text_norm], normalize_embeddings=True)[0]
            self.r.set(key, json.dumps(vec.tolist()))
            return vec
        else:
            return self.cached_embed_lru(text_norm)


# Global instance
embedding_service = EmbeddingService()