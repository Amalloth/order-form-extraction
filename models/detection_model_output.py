from typing import List

from pydantic import BaseModel

class ModelOutput(BaseModel):
    label: str
    score: float
    bbox: List[float] # Length 4