import pickle

import pandas as pd
import spotipy
from fastapi import FastAPI, HTTPException
from spotipy.oauth2 import SpotifyClientCredentials

# Configuration: environment variable to use in practical
CLIENT_ID = "Client_id"
CLIENT_SECRET = "Secret_client_id"

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))

# Model loading
app = FastAPI(title="Spotify Hit Predictor")

model = pickle.load(open("../training/model.pkl", "rb"))
model_columns = pickle.load(open("../training/columns.pkl", "rb"))
list_top5 = pickle.load(open("../training/top_5.pkl", "rb"))

classes = ["Not a Hit", "Hit"]
history = []

def fetch_features_from_spotify(track_input: str):
    if "open.spotify.com" in track_input or "track" in track_input:
        # ID extraction from link
        track_id = track_input.split("/")[-1].split("?")[0]
    else:
        # Search title from text
        results = sp.search(q=track_input, limit=1, type='track')
        if not results['tracks']['items']:
            return None

        track_id = results['tracks']['items'][0]['id']
        track_name = results['tracks']['items'][0]['name']

    audio_features = sp.audio_features(track_id)[0]
    print("Track name:", track_name)

    return audio_features


@app.get("/")
def home():
    return{"Homepage of Spotify Hit Predictor"} #   return render_template("homepage.html")


@app.get("/features")
def get_top_features():
    return {"top_5": list_top5}


# Predict with dictionary: not suitable for user
@app.post("/predict_dict")
def predict_dict(data: dict):
    df_input = pd.DataFrame([data], columns=model_columns)

    pred_index = int(model.predict(df_input)[0]) # index
    name = classes[pred_index]

    result = {
        "id": len(history),
        "input": data,
        "class_index": pred_index,
        "class_name": name
    }

    history.append(result)

    return result


@app.get("/past_predictions/{prediction_id}")
def get_prediction_by_id(prediction_id: int):

    if prediction_id < 0 or prediction_id >= len(history):
        raise HTTPException(status_code=404, detail="Prediction ID not found")

    return history[prediction_id]

@app.get("/health")
async def health():
    return {"status": "ok"}

# Predict with track name or direct Spotify link
@app.post("/predict")
def predict(data: dict):
    # Format {"query": "Daft Punk Get Lucky"}
    # or direct link {"query": "https://open.spotify.com/track/..."}
    query = data.get("query")

    if not query:
        raise HTTPException(status_code=400, detail="Put a title, or a link")

    spotify_data = fetch_features_from_spotify(query)

    if not spotify_data:
        raise HTTPException(status_code=404, detail="Title not found")

    input_dict = {col: spotify_data.get(col, 0) for col in model_columns}
    df_input = pd.DataFrame([input_dict], columns=model_columns)

    pred_index = int(model.predict(df_input)[0])
    name = classes[pred_index]

    result = {
        "id": len(history),
        "query": query,
        "class_name": name,
        "features_used": input_dict
    }
    history.append(result)

    return result


@app.get("/past_predictions")
def get_all_past_predictions():
    return history


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
