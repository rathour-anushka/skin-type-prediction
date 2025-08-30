from pydantic import BaseModel
from typing import Dict, Optional

class PredictionResponse(BaseModel):
    predicted_skin_type: str
    confidence: float
    lighting_ok: bool
    probabilities: Dict[str, float]

class Base64ImageRequest(BaseModel):
    image_base64: str
