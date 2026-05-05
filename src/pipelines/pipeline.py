from kfp import dsl

from src.pipelines.components.data_import import import_bigquery
from src.pipelines.components.evaluation import evaluate_model
from src.pipelines.components.preprocessing import training_preprocess
from src.pipelines.components.training import train_model
from src.pipelines.components.register_model import register_model
from src.pipelines.components.deploy_model import deploy_model
from src.pipelines.config import (
    PIPELINE_NAME,
    PROJECT_ID,
    REGION,
    MODEL_DISPLAY_NAME,
    ENDPOINT_DISPLAY_NAME,
    SERVING_CONTAINER_URI,
    ACCURACY_THRESHOLD,
)


@dsl.pipeline(name=PIPELINE_NAME)
def ai_dj_training_workflow(
    project_id: str = PROJECT_ID,
    model_version: str = "v1",
):
    # ------------------------------------------------------------------ #
    # Step 1 — Data ingestion from BigQuery                               #
    # ------------------------------------------------------------------ #
    data_task = import_bigquery(
        project_id=project_id,
    )

    # ------------------------------------------------------------------ #
    # Step 2 — Preprocessing → train / test split                        #
    # ------------------------------------------------------------------ #
    preprocess_task = training_preprocess(
        input_dataset=data_task.outputs["output_dataset"],
    )

    # ------------------------------------------------------------------ #
    # Step 3 — Model training (train split only)                         #
    # ------------------------------------------------------------------ #
    train_task = train_model(
        preprocessed_dataset=preprocess_task.outputs["train_dataset"],
    )

    # ------------------------------------------------------------------ #
    # Step 4 — Evaluation (test split only)                              #
    # ------------------------------------------------------------------ #
    evaluate_task = evaluate_model(
        model=train_task.outputs["model"],
        preprocessed_dataset=preprocess_task.outputs["test_dataset"],
    )

    # ------------------------------------------------------------------ #
    # Step 5 — Register model in Vertex AI Model Registry                #
    #          Includes quality gate: fails if accuracy < threshold       #
    # ------------------------------------------------------------------ #
    register_task = register_model(
        model=train_task.outputs["model"],
        metrics=evaluate_task.outputs["metrics"],
        project_id=project_id,
        location=REGION,
        model_display_name=MODEL_DISPLAY_NAME,
        serving_container_image_uri=SERVING_CONTAINER_URI,
        accuracy_threshold=ACCURACY_THRESHOLD,
    )

    # ------------------------------------------------------------------ #
    # Step 6 — Deploy model to Vertex AI Endpoint                        #
    # ------------------------------------------------------------------ #
    deploy_task = deploy_model(
        registered_model=register_task.outputs["registered_model"],
        project_id=project_id,
        location=REGION,
        endpoint_display_name=ENDPOINT_DISPLAY_NAME,
    )