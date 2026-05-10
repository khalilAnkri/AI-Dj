from kfp.dsl import Input, Metrics, Model, Output, component

from src.pipelines.config import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
    ],
)
def register_model(
    model: Input[Model],
    metrics: Input[Metrics],
    registered_model: Output[Model],
    project_id: str,
    location: str,
    model_display_name: str,
    serving_container_image_uri: str,
    accuracy_threshold: float = 0.75,
):
    """
    Uploads the trained model artifact to the Vertex AI Model Registry.
    Only registers if test_accuracy meets the threshold — acts as a
    quality gate before any deployment happens.
    """
    from google.cloud import aiplatform, storage

    # 1. Quality gate — read the test accuracy logged by evaluate_model
    test_accuracy = metrics.metadata.get("test_accuracy", 0.0)
    print(
        f"Quality gate check: test_accuracy={test_accuracy:.4f} (threshold={accuracy_threshold})"
    )

    if test_accuracy < accuracy_threshold:
        raise ValueError(
            f"Model did NOT pass quality gate. "
            f"test_accuracy={test_accuracy:.4f} < threshold={accuracy_threshold}. "
            f"Halting pipeline — model will NOT be registered or deployed."
        )

    print(f"Quality gate PASSED. Registering model '{model_display_name}'...")

    # 2. KFP saves the model

    source_uri = model.uri + ".pkl"
    bucket_name = source_uri.split("/")[2]
    source_blob = "/".join(source_uri.split("/")[3:])

    staging_blob = f"model_staging/{model_display_name}/model.pkl"
    staging_uri = f"gs://{bucket_name}/model_staging/{model_display_name}"

    print(f"Copying {source_uri}  →  gs://{bucket_name}/{staging_blob}")
    gcs_client = storage.Client(project=project_id)
    bucket = gcs_client.bucket(bucket_name)
    source = bucket.blob(source_blob)
    bucket.copy_blob(source, bucket, staging_blob)
    print("Copy complete. Staging URI:", staging_uri)

    # 3. Initialize Vertex AI SDK
    aiplatform.init(project=project_id, location=location)

    # 4. Upload to Model Registry — point at the clean staging folder
    vertex_model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=staging_uri,
        serving_container_image_uri=serving_container_image_uri,
        # Pass the feature column names stored by train_model as labels
        labels={
            "framework": "sklearn",
            "pipeline": "ai-dj-training-pipeline",
        },
    )

    print("Model registered successfully.")
    print(f"Resource name : {vertex_model.resource_name}")
    print(f"Model ID      : {vertex_model.name}")

    # 5. Pass model resource name downstream to the deploy component
    registered_model.metadata["resource_name"] = vertex_model.resource_name
    registered_model.metadata["model_id"] = vertex_model.name
    registered_model.metadata["display_name"] = model_display_name
    registered_model.metadata["test_accuracy"] = test_accuracy
