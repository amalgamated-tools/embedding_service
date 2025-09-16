from pydantic import BaseModel
from typing import List


class Item(BaseModel):
    text: str


class Items(BaseModel):
    texts: List[str]

    