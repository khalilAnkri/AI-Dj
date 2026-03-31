from fastapi import FastAPI, HTTPException
from .services import SpotifyService, PredictionService
from .config import Config

app = FastAPI(title="Spotify Hit Predictor (Vertex AI Edition)")


spotify_service = SpotifyService()
prediction_service = PredictionService()
history = []

@app.get("/health")
async def health():
    return {"status": "ok", "mode": "vertex_ai"}

@app.post("/predict")
async def predict(data: dict):
    query = data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query missing")

    # 1. Fetch from Spotify
    audio_features = spotify_service.get_features(query)
    if not audio_features:
        raise HTTPException(status_code=404, detail="Track not found")

    # 2. Get Prediction from Vertex AI
    try:
        result = prediction_service.predict(audio_features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vertex AI Error: {str(e)}")

    # 3. Format & Store
    response = {
        "id": len(history),
        "query": query,
        "prediction": result["class_name"],
        "confidence": result.get("confidence", "N/A") 
    }
    history.append(response)
    return response

@app.get("/past_predictions")
def get_history():
    return history


@app.get("/past_predictions/{prediction_id}")
def get_prediction_by_id(prediction_id: int):

    if prediction_id < 0 or prediction_id >= len(history):
        raise HTTPException(status_code=404, detail="Prediction ID not found")

    return history[prediction_id]


@app.post("/predict_manual")
async def predict_manual(data: dict):
    """
    Accepts a JSON dictionary of 13 audio features.
    Example: {"danceability": 0.8, "energy": 0.7, ...}
    """
    if not data:
        raise HTTPException(status_code=400, detail="JSON body is empty")

    try:
        # Pass the dictionary directly to the prediction service
        result = prediction_service.predict_manually(data)
        
        response = {
            "id": len(history),
            "source": "manual_input",
            "prediction": result["class_name"],
            "class_index": result["class_index"],
            "features_received": data
        }
        
        history.append(response)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vertex AI Error: {str(e)}")