from typing import List

from pydantic import BaseModel

class DetectedTable(BaseModel):
    label: str
    score: float
    bbox: List[float] # Length 4