from kfp import dsl

from src.pipelines.components.data_import import import_bigquery
from src.pipelines.components.evaluation import evaluate_model
from src.pipelines.components.preprocessing import training_preprocess
from src.pipelines.components.training import train_model
from src.pipelines.config import PIPELINE_NAME, PROJECT_ID


@dsl.pipeline(name=PIPELINE_NAME)
def ai_dj_training_workflow(
    project_id: str = PROJECT_ID,
    model_version: str = "v1"
):
    # Step 1: Data ingestion
    data_task = import_bigquery(project_id=project_id)

    # Step 2: Preprocessing (Now generates TWO outputs)
    preprocess_task = training_preprocess(
        input_dataset=data_task.outputs["output_dataset"]
    )

    # Step 3: Training (Only sees the "train" split)
    train_task = train_model(
        preprocessed_dataset=preprocess_task.outputs["train_dataset"]
    )

    # Step 4: Evaluation (Only sees the "test" split)
    evaluate_task = evaluate_model(
        model=train_task.outputs["model"],
        preprocessed_dataset=preprocess_task.outputs["test_dataset"]
    )
