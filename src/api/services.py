import spotipy
from google.cloud import aiplatform
from spotipy.oauth2 import SpotifyClientCredentials

from .config import Config


class SpotifyService:
    def __init__(self):
        cid = Config.SPOTIFY_CLIENT_ID
        secret = Config.SPOTIFY_CLIENT_SECRET

        print("Initializing SpotifyService:")

        self.auth_manager = SpotifyClientCredentials(
            client_id=cid,
            client_secret=secret
        )

        try:
            token = self.auth_manager.get_access_token(as_dict=False)
            print("Spotify token OK:", token[:20], "...")
        except Exception as e:
            raise Exception(f"Spotify AUTH FAILED: {e}")

        self.sp = spotipy.Spotify(auth_manager=self.auth_manager)

    def get_features(self, query: str):
        print(f"Query received: {query}")

        try:
            # Direct Spotify track URL
            if "spotify.com/track/" in query:
                track_id = query.split("track/")[-1].split("?")[0]
                print(f"Extracted track_id: {track_id}")

            # Search by name
            else:
                print("Searching track on Spotify:")
                results = self.sp.search(q=query, limit=1, type='track')

                if not results or not results.get('tracks', {}).get('items'):
                    raise ValueError("Track not found")

                track = results['tracks']['items'][0]
                track_id = track['id']

                print(f"Found track: {track['name']} - {track['artists'][0]['name']}")

            # Get audio features
            print("Fetching audio features:")
            features = self.sp.audio_features(track_id)

            if not features or features[0] is None:
                raise ValueError("Audio features not available")

            print("Features retrieved")
            return features[0]

        except ValueError as ve:
            # Expected errors (clean API response)
            return {"error": str(ve)}

        except Exception as e:
            # REAL errors (auth, network, etc.)
            print(f"Spotify API ERROR: {e}")
            raise Exception(f"Spotify API failure: {e}")

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
