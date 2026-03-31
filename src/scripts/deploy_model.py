from google.cloud import aiplatform
import os
from src.api.config import Config


# Configuration
PROJECT_ID = Config.PROJECT_ID
LOCATION = Config.LOCATION
MODEL_ID = Config.MODEL_ID

def deploy():
    aiplatform.init(project=PROJECT_ID, location=LOCATION)

    # 1. Get the model from the registry
    model = aiplatform.Model(f"projects/{PROJECT_ID}/locations/{LOCATION}/models/{MODEL_ID}")

    print(f"Deploying model {MODEL_ID} to endpoint in {LOCATION}...")
    print("Note: This will take 10-15 minutes. Google is booting up the server.")

    # 2. Deploy
    endpoint = model.deploy(
        deployed_model_display_name="spotify_hit_predictor_live",
        machine_type="n1-standard-2",
        min_replica_count=1, 
        max_replica_count=2,
    )

    print("-" * 30)
    print("SUCCESS: Our model is live!")
    print(f"ENDPOINT_ID: {endpoint.name}")
    print("-" * 30)

if __name__ == "__main__":
    deploy()