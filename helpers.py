import re

def normalize(text: str) -> str:
    """Normalize text for consistent embedding"""
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # remove numbers
    text = re.sub(r'[^a-z ]+', '', text)  # remove punctuation
    return text.strip()