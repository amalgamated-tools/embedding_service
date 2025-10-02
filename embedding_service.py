import os
import helpers
import logging
from categories import Categories
from schemas import ClassifyResponse
from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np # type: ignore
import redis # type: ignore
import json
from functools import lru_cache

logging.basicConfig(level=logging.DEBUG)

class EmbeddingService:
    def __init__(self):
        # Configuration
        self.model_name = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")
        self.redis_host = os.getenv("REDIS_HOST", "redis")
        self.redis_port = int(os.getenv("REDIS_PORT", 6379))
        
        # Load model
        logging.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        self.categories = Categories()
        
        # Anchors - Major department categories
        self.anchors = [
            "account executive",
            "account manager",
            "agent engineering",
            "agentic platform",
            "agents",
            "ai engineer",
            "ai researcher",
            "ai",
            "ai/ml",
            "analytics",
            "applied ai infrastructure",
            "backend developer",
            "backend engineering",
            "backend", 
            "brand manager",
            "brand",
            "business development manager",
            "business development",
            "business",
            "client success",
            "client",
            "commercial",
            "community manager",
            "community relations",
            "community",
            "company",
            "content designer",
            "content marketing manager",
            "content marketing",
            "content strategist",
            "copywriter",
            "corporate it",
            "customer experience",
            "customer success manager",
            "customer success",
            "customer support",
            "cybersecurity",
            "data analyst",
            "data engineer",
            "data engineering",
            "data science",
            "data scientist",
            "data",
            "design researcher",
            "design",
            "desktop support",
            "developer advocate", 
            "developer evangelist",
            "developer relations",
            "development",
            "devops engineer",
            "devops",
            "digital marketing",
            "ecosystem",
            "education",
            "electrical engineer",
            "electrical",
            "embedded engineer",
            "embeddings",
            "engineering",
            "enterprise",
            "executive",
            "facilities",
            "field",
            "finance",
            "firmware engineer",
            "firmware",
            "forward",
            "frontend developer",
            "frontend engineering",
            "frontend",
            "full stack developer",
            "full stack engineering",
            "full-stack",
            "fullstack",
            "gotomarket",
            "graphic designer",
            "grc",
            "growth manager",
            "growth",
            "gtm",
            "hardware engineer",
            "hardware",
            "helpdesk",
            "human resources",
            "incident response",
            "inference",
            "information security",
            "infosec",
            "infrastructure engineer",
            "infrastructure",
            "interaction designer",
            "it manager",
            "it operations",
            "it support",
            "large language model",
            "large language models",
            "legal",
            "llm",
            "machine learning engineer",
            "machine learning team",
            "machine learning",
            "marketing manager",
            "marketing specialist",
            "marketing",
            "mechanical engineer",
            "mechanical", 
            "ml engineer",
            "ml",
            "mobile developer",
            "mobile",
            "model training",
            "onboarding", 
            "operations",
            "people",
            "performance marketing",
            "performance",
            "platform engineer",
            "pm",
            "principal product manager",
            "privacy",
            "product designer",
            "product engineering",
            "product management",
            "product manager",
            "product marketing manager",
            "product marketing",
            "product owner",
            "program manager",
            "qa engineer",
            "qa",
            "qa/test",
            "research scientist",
            "research",
            "robotics engineer",
            "robotics",
            "sales associate",
            "sales engineer",
            "sales manager",
            "sales representative",
            "sales",
            "security analyst",
            "security engineer",
            "security engineering",
            "security operations",
            "security",
            "senior product manager",
            "site reliability",
            "software development",
            "software engineer",
            "software engineering",
            "solutions engineer",
            "sre",
            "support engineer",
            "support engineering",
            "system administrator",
            "talent",
            "technical account manager",
            "technical account mgmt",
            "technical pm",
            "technical product manager",
            "technical support",
            "technical writer",
            "technology",
            "test engineer",
            "ui designer",
            "user experience",                     
            "user researcher",
            "ux designer",
            "ux researcher",
            "ux writer",
            "visual designer",
        ]
        self.anchor_vecs = self.model.encode(self.anchors, normalize_embeddings=True)
        
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

    def check_anchor_match(self, text: str) -> str | None:
        """Check for direct string matches in the anchor list"""
        logging.info(f"Checking anchor match for: {text}")

        for key in self.anchors:
            if key.lower() == text.lower():
                logging.info(f"Anchor match found: {key} in '{text}'")
                return key

        logging.info(f"No anchor match found for '{text}'")
        return None

    def embed_or_cache(self, text: str):
        """Get embedding with Redis or LRU cache"""
        text_norm = helpers.normalize(text)
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
        
    def classify_batch(self, texts: list[str], threshold: float) -> list[ClassifyResponse]:
        results = []
        for text in texts:
            results.append(self.classify(text, threshold))
        return results

    def classify(self, text: str, threshold: float = 0.25) -> ClassifyResponse:
        logging.info(f"Classifying text: {text}")
        
        # check for direct anchor match first
        direct_match = self.check_anchor_match(text)
        if direct_match:
            mapped_category = self.categories.map_category(direct_match)
            logging.info(f"Direct match found: {direct_match} mapped to {mapped_category}")

            return ClassifyResponse(
                category_before_mapping=direct_match,
                mapped_category=mapped_category,
                classify_method="direct_match",
                similarity=1.0,
                input_text=text,
            )

        # check for literal string match in categories
        literal_match = self.categories.check_variant_match(text)
        if literal_match:
            mapped_category = self.categories.map_category(literal_match)
            logging.info(f"Literal match found: {literal_match} mapped to {mapped_category}")

            return ClassifyResponse(
                category_before_mapping=literal_match,
                mapped_category=mapped_category,
                classify_method="literal_match",
                similarity=1.0,
                input_text=text,
            )

        # no direct match, proceed with embedding-based classification
        text_vec = self.embed_or_cache(text)
        sims = np.dot(self.anchor_vecs, text_vec)

        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        best_anchor = self.anchors[best_idx]
        mapped_category = self.categories.map_category(best_anchor)

        logging.info(f"Best anchor: {best_anchor} with similarity {best_sim} mapped to {mapped_category}")

        if best_sim < threshold:
            logging.info(f"Similarity {best_sim} below threshold {threshold}. Assigning 'Other'.")
            mapped_category = "unsure"

        return ClassifyResponse(
            mapped_category=mapped_category,
            category_before_mapping=best_anchor,
            closest_anchor=best_anchor,
            classify_method="embedding",
            similarity=float(best_sim),
            input_text=text,
        )


# Global instance
embedding_service = EmbeddingService()