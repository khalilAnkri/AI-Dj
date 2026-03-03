from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI-DJ Backend")

class MusicFeatures(BaseModel):
    danceability: float
    energy: float
    tempo: float

@app.get("/")
def health():
    return {"status": "healthy", "service": "api"}

@app.post("/predict")
def predict(data: MusicFeatures):
    return {
        "prediction": "Summer Hit",
        "confidence": 0.95,
        "input_received": data
    }