from datetime import UTC, datetime

import asyncio

from fastapi import FastAPI, HTTPException

from .services import (
    ExplanationService,
    PredictionService,
    RecommendationService,
    SpotifyService,
)

INTER_REQUEST_DELAY = 0.5  # seconds between Musicae calls to avoid burst-rate 429s

app = FastAPI(title="Spotify Hit Predictor")

spotify_service = SpotifyService()
prediction_service = PredictionService()
explanation_service = ExplanationService()
recommendation_service = RecommendationService()

history: list = []


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "mode": "vertex-ai"}


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

@app.post("/predict")
async def predict(data: dict):
    query = data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query missing")

    # 1 — Audio features (Musicae call #1)
    audio_features = spotify_service.get_features(query)
    if not audio_features:
        raise HTTPException(status_code=404, detail="Track not found")
    if "error" in audio_features:
        raise HTTPException(status_code=404, detail=audio_features["error"])

    track_id = audio_features.pop("_track_id", None)

    # Small pause before next Musicae call to avoid burst-rate 429s
    await asyncio.sleep(INTER_REQUEST_DELAY)

    # 2 — Track metadata (Musicae call #2)
    metadata   = spotify_service.get_metadata(track_id) if track_id else {}
    track_name = metadata.get("track_name", "Unknown")
    artist = metadata.get("artist", "Unknown")

    # 3 — ML prediction (Vertex AI — not subject to Musicae rate limits)
    try:
        result = prediction_service.predict(audio_features)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        ) from e

    # 4 — Explanation (local, no API call)
    explanation = explanation_service.explain(
        track_name=track_name,
        artist=artist,
        prediction=result["class_name"],
        confidence=result["confidence"],
        hit_probability=1.0 if result["class_index"] == 1 else 0.0,
        top_features=result["top_features"],
    )

    # Small pause before next Musicae call to avoid burst-rate 429s
    await asyncio.sleep(INTER_REQUEST_DELAY)

    # 5 — Recommendations (Musicae call #3)
    artist_id       = metadata.get("artist_id", "")
    recommendations = (
        recommendation_service.get_recommendations(track_id, artist_id, audio_features)
        if track_id else []
    )

    response = {
        "id":                len(history),
        "predicted_at":      datetime.now(timezone.utc).isoformat(),
        "query":             query,
        "track_name":        track_name,
        "artist":            artist,
        "thumbnail":         metadata.get("thumbnail", ""),
        "spotify_url":       metadata.get("spotify_url", query),
        "prediction":        result["class_name"],
        "confidence":        f"{round(result['confidence'] * 100, 1)}%",
        "top_features":      result["top_features"],
        "explanation":       explanation,
        "if_you_liked_this": recommendations,
    }

    history.append(response)
    return response


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@app.get("/past_predictions")
def get_history():
    return history


@app.get("/past_predictions/{prediction_id}")
def get_prediction_by_id(prediction_id: int):
    if prediction_id < 0 or prediction_id >= len(history):
        raise HTTPException(status_code=404, detail="Prediction ID not found")
    return history[prediction_id]


# ---------------------------------------------------------------------------
# Manual predict
# ---------------------------------------------------------------------------

@app.post("/predict_manual")
async def predict_manual(data: dict):
    if not data:
        raise HTTPException(status_code=400, detail="JSON body is empty")
    try:
        result = prediction_service.predict_manually(data)
        explanation = explanation_service.explain(
            track_name="Manual Input",
            artist="N/A",
            prediction=result["class_name"],
            confidence=result["confidence"],
            hit_probability=1.0 if result["class_index"] == 1 else 0.0,
            top_features=result["top_features"],
        )
        response = {
            "id": len(history),
            "predicted_at": datetime.now(UTC).isoformat(),
            "source": "manual_input",
            "prediction": result["class_name"],
            "confidence": f"{round(result['confidence'] * 100, 1)}%",
            "top_features": result["top_features"],
            "explanation": explanation,
            "features_received": data,
        }
        history.append(response)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        ) from e
