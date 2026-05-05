import os

from dotenv import load_dotenv

# This looks for the .env file and loads the variables
load_dotenv()

class Config:
    PROJECT_ID              = os.getenv("GCP_PROJECT_ID")
    LOCATION                = os.getenv("GCP_LOCATION")
    ENDPOINT_ID             = os.getenv("VERTEX_ENDPOINT_ID")
    BUCKET_NAME             = os.getenv("GCP_BUCKET_NAME")
    BUCKET_URI              = os.getenv("GCP_BUCKET_URI")
    GCS_MODEL_ARTIFACTS_URI = os.getenv("GCS_MODEL_ARTIFACTS_URI")
    MODEL_ID                = os.getenv("MODEL_ID")
    RAPIDAPI_KEY            = os.getenv("RAPIDAPI_KEY")
 
    CLASSES       = ["Flop", "Hit"]
    MODEL_COLUMNS = [
        "duration_ms", "danceability", "energy", "key", "loudness",
        "mode", "speechiness", "acousticness", "instrumentalness",
        "liveness", "valence", "tempo", "time_signature"
    ]
