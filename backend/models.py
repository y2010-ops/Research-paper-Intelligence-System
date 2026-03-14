from pydantic import BaseModel
from typing import List, Optional

class Paper(BaseModel):
    title: str
    authors: List[str]
    summary: str
    url: str

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5
