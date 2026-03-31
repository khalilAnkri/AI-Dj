import os
from dotenv import load_dotenv

# This looks for the .env file and loads the variables
load_dotenv() 

class Config:
    CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
    CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
    PROJECT_ID = os.getenv("GCP_PROJECT_ID")
    LOCATION = os.getenv("GCP_LOCATION")
    ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID")
    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    BUCKET_URI = os.getenv("GCP_BUCKET_URI")
    GCS_MODEL_ARTIFACTS_URI = os.getenv("GCS_MODEL_ARTIFACTS_URI")
    MODEL_ID = os.getenv("MODEL_ID")
    

    CLASSES = ["Flop", "Hit"]
    MODEL_COLUMNS = [
            "danceability", "energy", "key", "loudness", "mode", 
            "speechiness", "acousticness", "instrumentalness", 
            "liveness", "valence", "tempo", 
            "duration_ms", "time_signature"  
        ]