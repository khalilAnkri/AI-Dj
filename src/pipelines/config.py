import os

from dotenv import load_dotenv

load_dotenv()


# --- GCP Core ---
BASE_IMAGE = "python:3.10"
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
REGION = os.getenv("GCP_LOCATION")

# --- GCS Bucket ---
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
BUCKET_URI = os.getenv("GCP_BUCKET_URI", f"gs://{BUCKET_NAME}")


PIPELINE_NAME = "ai-dj-training-pipeline"


MODEL_DISPLAY_NAME = "spotify-hit-predictor"


SERVING_CONTAINER_URI = (
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-4:latest"
)


GCS_MODEL_ARTIFACTS_URI = os.getenv(
    "GCS_MODEL_ARTIFACTS_URI", f"gs://{BUCKET_NAME}/pipeline_root"
)


ENDPOINT_DISPLAY_NAME = "spotify-hit-predictor-endpoint"


ACCURACY_THRESHOLD = 0.75
