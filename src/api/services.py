import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from google.cloud import aiplatform
from .config import Config

class SpotifyService:
    def __init__(self):
        auth_manager = SpotifyClientCredentials(
            client_id=Config.CLIENT_ID, 
            client_secret=Config.CLIENT_SECRET
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def get_features(self, query: str):
        # --- MOCK LOGIC FOR TESTING ---
        if query == "MOCK_HIT":
            return {
                "danceability": 0.8, "energy": 0.9, "key": 1, "loudness": -4.0,
                "mode": 1, "speechiness": 0.05, "acousticness": 0.1,
                "instrumentalness": 0.0, "liveness": 0.2, "valence": 0.9, "tempo": 125,
                "duration_ms": 210000, "time_signature": 4 
            }
        
        if query == "MOCK_FLOP":
            return {
                "danceability": 0.2, "energy": 0.1, "key": 0, "loudness": -25.0,
                "mode": 0, "speechiness": 0.9, "acousticness": 0.9,
                "instrumentalness": 0.9, "liveness": 0.8, "valence": 0.1, "tempo": 60,
                "duration_ms": 180000, "time_signature": 3
            }
        # --- END MOCK LOGIC ---

        try:
            # Check if it's a direct URL
            if "spotify.com" in query:
                track_id = query.split("/")[-1].split("?")[0]
            else:
                results = self.sp.search(q=query, limit=1, type='track')
                if not results['tracks']['items']:
                    return None
                track_id = results['tracks']['items'][0]['id']
            
            features = self.sp.audio_features(track_id)
            return features[0] if features else None
            
        except Exception as e:
            # If Spotify fails (like the 403 error), we catch it here
            print(f"Spotify API Error: {e}")
            return None

class PredictionService:
    def __init__(self):
        # Connect to  Live Google Cloud Endpoint
        aiplatform.init(project=Config.PROJECT_ID, location=Config.LOCATION)
        self.endpoint = aiplatform.Endpoint(Config.ENDPOINT_ID)

    def predict(self, features: dict):
        # 1. Extract features in the EXACT order the model was trained on
        try:
            input_data = [float(features.get(col, 0)) for col in Config.MODEL_COLUMNS]
            
            # 2. Call Vertex AI Endpoint
            prediction = self.endpoint.predict(instances=[input_data])
            
            # 3. Parse the result (Vertex returns floats, we want the class index)
            pred_index = int(prediction.predictions[0])
            
            return {
                "class_index": pred_index,
                "class_name": Config.CLASSES[pred_index],
                "raw_prediction": prediction.predictions[0]
            }
        except Exception as e:
            print(f"Vertex AI Prediction Error: {e}")
            raise e
    

    def predict_manually(self, features: dict):
        try:
            input_data = [
                float(features.get("danceability", 0)),
                float(features.get("energy", 0)),
                float(features.get("key", 0)),
                float(features.get("loudness", 0)),
                float(features.get("mode", 0)),
                float(features.get("speechiness", 0)),
                float(features.get("acousticness", 0)),
                float(features.get("instrumentalness", 0)),
                float(features.get("liveness", 0)),
                float(features.get("valence", 0)),
                float(features.get("tempo", 0)),
                float(features.get("duration_ms", 0)),   
                float(features.get("time_signature", 0)) 
            ]
            
            # Call Vertex AI 
            prediction = self.endpoint.predict(instances=[input_data])
            pred_index = int(prediction.predictions[0])
            
            return {
                "class_index": pred_index,
                "class_name": Config.CLASSES[pred_index]
            }
        except Exception as e:
            print(f"Prediction logic error: {e}")
            raise e