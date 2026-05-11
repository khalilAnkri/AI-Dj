from google.cloud import aiplatform

from src.api.config import Config

PROJECT_ID = Config.PROJECT_ID
LOCATION = Config.LOCATION
GCP_BUCKET_NAME = Config.BUCKET_NAME
GCP_BUCKET_URI = Config.BUCKET_URI


# 1. Initialize explicitly for  region
aiplatform.init(project=PROJECT_ID, location=LOCATION, staging_bucket=GCP_BUCKET_URI)

# 2. Use the EUROPE-specific pre-built container URI
EUROPE_CONTAINER_URI = (
    "europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-4:latest"
)

GCS_MODEL_ARTIFACTS_URI = Config.GCS_MODEL_ARTIFACTS_URI

model = aiplatform.Model.upload(
    display_name="spotify-hit-predictor",
    artifact_uri=GCS_MODEL_ARTIFACTS_URI,
    serving_container_image_uri=EUROPE_CONTAINER_URI,
)

print(f"Model registered  , Resource Name: {model.resource_name}")
