import logging
import helpers

logging.basicConfig(level=logging.DEBUG)

class Categories:
    def __init__(self):
        self.categories = [
            "AI",
            "Corporate IT",
            "Customer Success / Support", 
            "Community", 
            "Data",
            "Design",
            "Hardware", 
            "Marketing",
            "Product Management",
            "Sales",
            "Security",
            "Software Engineering", 
        ]        

        self.variant_map = {
            # AI mappings
            "agent engineering": "AI",
            "agentic platform": "AI",
            "agents": "AI",
            "ai engineer": "AI",
            "ai researcher": "AI",
            "ai/ml": "AI",
            "aiml": "AI",
            "applied ai infrastructure": "AI",
            "embeddings": "AI",
            "inference": "AI",
            "large language model": "AI",
            "large language models": "AI",
            "llm": "AI",
            "machine learning engineer": "AI",
            "machine learning team": "AI",
            "machine learning": "AI",
            "ml engineer": "AI",
            "ml": "AI",
            "model training": "AI",
            "research scientist": "AI",
            "research": "AI",

            # Corporate IT mappings
            "corporate it": "Corporate IT",
            "helpdesk": "Corporate IT",
            "it support": "Corporate IT",
            "it manager": "Corporate IT",
            "system administrator": "Corporate IT",
            "it operations": "Corporate IT",
            "desktop support": "Corporate IT",

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
            "technical support": "Customer Success / Support",
            
            # Community mappings
            "developer relations": "Community",
            "developer advocate": "Community", 
            "community manager": "Community",
            "developer evangelist": "Community",
            "community relations": "Community",
                        
            # Data & AI mappings
            "analytics": "Data",
            "data analyst": "Data",
            "data engineer": "Data",
            "data engineering": "Data",
            "data science": "Data",
            "data scientist": "Data",

            # Design mappings
            "content designer": "Design",
            "content strategist": "Design",
            "copywriter": "Design",
            "design researcher": "Design",
            "graphic designer": "Design",
            "interaction designer": "Design",
            "product designer": "Design",
            "technical writer": "Design",
            "ui designer": "Design",
            "user researcher": "Design",
            "ux designer": "Design",
            "ux researcher": "Design",
            "ux writer": "Design",
            "visual designer": "Design",
            "user experience": "Design",                     

            # Hardware mappings
            "electrical engineer": "Hardware",
            "electrical": "Hardware",
            "embedded engineer": "Hardware",
            "firmware engineer": "Hardware",
            "firmware": "Hardware",
            "hardware engineer": "Hardware",
            "mechanical engineer": "Hardware",
            "mechanical": "Hardware", 
            "robotics engineer": "Hardware",
            "robotics": "Hardware",
          
            # Marketing mappings
            "brand manager": "Marketing",
            "brand": "Marketing",
            "content marketing manager": "Marketing",
            "content marketing": "Marketing",
            "digital marketing": "Marketing",
            "gotomarket": "Marketing",
            "growth manager": "Marketing",
            "growth": "Marketing",
            "gtm": "Marketing",
            "marketing manager": "Marketing",
            "marketing specialist": "Marketing",
            "performance marketing": "Marketing",
            "performance": "Marketing",
            "product marketing manager": "Marketing",
            "product marketing": "Marketing",
            
            # Product Management mappings
            "technical pm": "Product Management",
            "pm": "Product Management",
            "program manager": "Product Management",
            "product owner": "Product Management",
            "product manager": "Product Management",
            "technical product manager": "Product Management",
            "senior product manager": "Product Management",
            "principal product manager": "Product Management",
                        
            # Sales mappings
            "account executive": "Sales",
            "account manager": "Sales",
            "business development manager": "Sales",
            "business development": "Sales",
            "sales associate": "Sales",
            "sales engineer": "Sales",
            "sales manager": "Sales",
            "sales representative": "Sales",
            "solutions engineer": "Sales",
            
            # Security mappings
            "security engineering": "Security",
            "grc": "Security",
            "incident response": "Security",
            "security analyst": "Security",
            "privacy": "Security",
            "cybersecurity": "Security",
            "information security": "Security",
            "security operations": "Security",
            "infosec": "Security",

            # Software Engineering mappings
            "backend developer": "Software Engineering",
            "backend": "Software Engineering", 
            "backend engineering": "Software Engineering",
            "devops engineer": "Software Engineering",
            "devops": "Software Engineering",
            "frontend developer": "Software Engineering",
            "frontend": "Software Engineering",
            "full stack developer": "Software Engineering",
            "full-stack": "Software Engineering",
            "fullstack": "Software Engineering",
            "infrastructure engineer": "Software Engineering",
            "infrastructure": "Software Engineering",
            "mobile developer": "Software Engineering",
            "mobile": "Software Engineering",
            "platform engineer": "Software Engineering",
            "product engineering": "Software Engineering",
            "qa engineer": "Software Engineering",
            "qa": "Software Engineering",
            "qa/test": "Software Engineering",
            "security engineer": "Software Engineering",
            "security": "Software Engineering",
            "site reliability": "Software Engineering",
            "software development": "Software Engineering",
            "software engineer": "Software Engineering",
            "software engineering": "Software Engineering",
            "sre": "Software Engineering",
            "test engineer": "Software Engineering",
        }        

    def map_category(self, category: str) -> str:
        # Return the mapped category, or the original category if no mapping exists
        # This allows anchor categories to pass through unchanged while mapping specific job titles
        logging.info(f"Mapping category for: {category}")
        category =  helpers.normalize(category)
        logging.info(f"Normalized category: {category}")
        mapped = self.variant_map.get(category, category)
        logging.info(f"Mapped category: {mapped}")

        return mapped        

    def check_variant_match(self, text: str) -> str | None:
        """Check for direct string matches in the variant map"""
        logging.info(f"Checking literal string match for: {text}")
        
        # Also check normalized variants
        text_norm = helpers.normalize(text)
        for key in self.variant_map.keys():
            if key == text_norm:
                logging.info(f"Literal string match found in variant map: {key} in '{text_norm}' with '{self.variant_map[key]}'")
                return self.variant_map[key]

        logging.info(f"No direct match found for '{text}' in variant map")
        return None        