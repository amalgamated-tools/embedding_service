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
        
        # Anchors - Major department categories
        self.anchors = [
            "Software Engineering",
            "Data & AI",
            "Hardware / Embedded",
            "Product Management",
            "Product / UX Design",
            "UI / Visual Design", 
            "UX Research",
            "Content / UX Writing",
            "Sales",
            "Marketing",
            "Customer Success / Support",
            "Community & Developer Relations",
            "People / HR / Recruiting / Talent",
            "Finance & Accounting",
            "Legal & Compliance",
            "Operations / Strategy / BizOps",
            "Facilities / Workplace Experience",
            "Corporate IT / Helpdesk",
            "Security & Privacy",
            "Executive roles",
        ]
        self.anchor_vecs = self.model.encode(self.anchors, normalize_embeddings=True)

        self.variant_map = {
            # Software Engineering mappings
            "frontend": "Software Engineering",
            "backend": "Software Engineering", 
            "full-stack": "Software Engineering",
            "mobile": "Software Engineering",
            "infrastructure": "Software Engineering",
            "security": "Software Engineering",
            "qa/test": "Software Engineering",
            "devops": "Software Engineering",
            "sre": "Software Engineering",
            "site reliability": "Software Engineering",
            "software engineer": "Software Engineering",
            "frontend developer": "Software Engineering",
            "backend developer": "Software Engineering",
            "full stack developer": "Software Engineering",
            "mobile developer": "Software Engineering",
            "infrastructure engineer": "Software Engineering",
            "security engineer": "Software Engineering",
            "qa engineer": "Software Engineering",
            "test engineer": "Software Engineering",
            "devops engineer": "Software Engineering",
            "platform engineer": "Software Engineering",
            "software development": "Software Engineering",
            
            # Data & AI mappings
            "data engineering": "Data & AI",
            "data science": "Data & AI",
            "machine learning": "Data & AI",
            "analytics": "Data & AI",
            "research": "Data & AI",
            "data engineer": "Data & AI",
            "data scientist": "Data & AI",
            "ml engineer": "Data & AI",
            "machine learning engineer": "Data & AI",
            "ai engineer": "Data & AI",
            "data analyst": "Data & AI",
            "research scientist": "Data & AI",
            "ai researcher": "Data & AI",
            
            # Hardware / Embedded mappings
            "electrical": "Hardware / Embedded",
            "mechanical": "Hardware / Embedded", 
            "firmware": "Hardware / Embedded",
            "robotics": "Hardware / Embedded",
            "electrical engineer": "Hardware / Embedded",
            "mechanical engineer": "Hardware / Embedded",
            "firmware engineer": "Hardware / Embedded",
            "embedded engineer": "Hardware / Embedded",
            "robotics engineer": "Hardware / Embedded",
            "hardware engineer": "Hardware / Embedded",
            
            # Product Management mappings
            "technical pm": "Product Management",
            "program manager": "Product Management",
            "product owner": "Product Management",
            "product manager": "Product Management",
            "technical product manager": "Product Management",
            "senior product manager": "Product Management",
            "principal product manager": "Product Management",
            
            # Design mappings
            "ux designer": "Product / UX Design",
            "product designer": "Product / UX Design",
            "interaction designer": "Product / UX Design",
            "ui designer": "UI / Visual Design",
            "visual designer": "UI / Visual Design",
            "graphic designer": "UI / Visual Design",
            "ux researcher": "UX Research",
            "user researcher": "UX Research",
            "design researcher": "UX Research",
            "content designer": "Content / UX Writing",
            "ux writer": "Content / UX Writing",
            "content strategist": "Content / UX Writing",
            "technical writer": "Content / UX Writing",
            
            # Sales mappings
            "account executive": "Sales",
            "business development": "Sales",
            "solutions engineer": "Sales",
            "sales engineer": "Sales",
            "account manager": "Sales",
            "business development manager": "Sales",
            "sales representative": "Sales",
            "sales manager": "Sales",
            
            # Marketing mappings
            "growth": "Marketing",
            "brand": "Marketing",
            "performance": "Marketing",
            "product marketing": "Marketing",
            "content marketing": "Marketing",
            "growth manager": "Marketing",
            "brand manager": "Marketing",
            "performance marketing": "Marketing",
            "product marketing manager": "Marketing",
            "content marketing manager": "Marketing",
            "marketing manager": "Marketing",
            "digital marketing": "Marketing",
            "marketing specialist": "Marketing",
            
            # Customer Success / Support mappings
            "support engineering": "Customer Success / Support",
            "onboarding": "Customer Success / Support", 
            "technical account mgmt": "Customer Success / Support",
            "customer success": "Customer Success / Support",
            "customer support": "Customer Success / Support",
            "support engineer": "Customer Success / Support",
            "customer success manager": "Customer Success / Support",
            "technical account manager": "Customer Success / Support",
            "customer experience": "Customer Success / Support",
            "client success": "Customer Success / Support",
            
            # Community & Developer Relations mappings
            "developer relations": "Community & Developer Relations",
            "developer advocate": "Community & Developer Relations", 
            "community manager": "Community & Developer Relations",
            "developer evangelist": "Community & Developer Relations",
            "community relations": "Community & Developer Relations",
            
            # People / HR / Recruiting / Talent mappings
            "people": "People / HR / Recruiting / Talent",
            "hr": "People / HR / Recruiting / Talent",
            "recruiting": "People / HR / Recruiting / Talent",
            "talent": "People / HR / Recruiting / Talent",
            "human resources": "People / HR / Recruiting / Talent",
            "recruiter": "People / HR / Recruiting / Talent",
            "talent acquisition": "People / HR / Recruiting / Talent",
            "hr manager": "People / HR / Recruiting / Talent",
            "people operations": "People / HR / Recruiting / Talent",
            "talent manager": "People / HR / Recruiting / Talent",
            
            # Finance & Accounting mappings
            "finance": "Finance & Accounting",
            "accounting": "Finance & Accounting",
            "financial analyst": "Finance & Accounting",
            "accountant": "Finance & Accounting",
            "finance manager": "Finance & Accounting",
            "controller": "Finance & Accounting",
            "treasury": "Finance & Accounting",
            "financial planning": "Finance & Accounting",
            
            # Legal & Compliance mappings
            "legal": "Legal & Compliance",
            "compliance": "Legal & Compliance",
            "lawyer": "Legal & Compliance",
            "attorney": "Legal & Compliance",
            "legal counsel": "Legal & Compliance",
            "compliance manager": "Legal & Compliance",
            "regulatory affairs": "Legal & Compliance",
            "legal operations": "Legal & Compliance",
            
            # Operations / Strategy / BizOps mappings
            "operations": "Operations / Strategy / BizOps",
            "strategy": "Operations / Strategy / BizOps", 
            "bizops": "Operations / Strategy / BizOps",
            "business operations": "Operations / Strategy / BizOps",
            "operations manager": "Operations / Strategy / BizOps",
            "strategy manager": "Operations / Strategy / BizOps",
            "business analyst": "Operations / Strategy / BizOps",
            "program operations": "Operations / Strategy / BizOps",
            
            # Facilities / Workplace Experience mappings
            "facilities": "Facilities / Workplace Experience",
            "workplace experience": "Facilities / Workplace Experience",
            "office manager": "Facilities / Workplace Experience",
            "facilities manager": "Facilities / Workplace Experience",
            "workplace operations": "Facilities / Workplace Experience",
            "office operations": "Facilities / Workplace Experience",
            
            # Corporate IT / Helpdesk mappings
            "corporate it": "Corporate IT / Helpdesk",
            "helpdesk": "Corporate IT / Helpdesk",
            "it support": "Corporate IT / Helpdesk",
            "it manager": "Corporate IT / Helpdesk",
            "system administrator": "Corporate IT / Helpdesk",
            "it operations": "Corporate IT / Helpdesk",
            "desktop support": "Corporate IT / Helpdesk",
            
            # Security & Privacy mappings
            "security engineering": "Security & Privacy",
            "grc": "Security & Privacy",
            "incident response": "Security & Privacy",
            "security analyst": "Security & Privacy",
            "privacy": "Security & Privacy",
            "cybersecurity": "Security & Privacy",
            "information security": "Security & Privacy",
            "security operations": "Security & Privacy",
            
            # Executive roles mappings
            "ceo": "Executive roles",
            "cto": "Executive roles",
            "vp eng": "Executive roles",
            "cpo": "Executive roles",
            "cmo": "Executive roles",
            "chief executive officer": "Executive roles",
            "chief technology officer": "Executive roles",
            "chief product officer": "Executive roles",
            "chief marketing officer": "Executive roles",
            "vp": "Executive roles",
            "vice president": "Executive roles",
            "director": "Executive roles",
            "head of": "Executive roles",
            "founder": "Executive roles",
            "co-founder": "Executive roles",
            "president": "Executive roles",
            
            # Legacy mappings for backward compatibility
            "Product Engineering": "Software Engineering",
            "Platform Engineering": "Software Engineering",
            "Software Engineering": "Software Engineering",
            "Product Manager": "Product Management",
            "UX Designer": "Product / UX Design",
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
        # Return the mapped category, or the original category if no mapping exists
        # This allows anchor categories to pass through unchanged while mapping specific job titles
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