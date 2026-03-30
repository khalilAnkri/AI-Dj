from kfp import dsl
from src.pipelines.components.data_import import import_bigquery
from src.pipelines.components.preprocessing import training_preprocess
from src.pipelines.components.training import train_model
from src.pipelines.components.evaluation import evaluate_model
from src.pipelines.config import PIPELINE_NAME, PROJECT_ID

@dsl.pipeline(name=PIPELINE_NAME)
def ai_dj_training_workflow(
    project_id: str = PROJECT_ID,
    # Adding a version or timestamp parameter here helps with Experiment Tracking
    model_version: str = "v1" 
):
    # Step 1: Data ingestion
    data_task = import_bigquery(project_id=project_id)

    # Step 2: Preprocessing
    preprocess_task = training_preprocess(
        input_dataset=data_task.outputs["output_dataset"]
    )

    # Step 3: Training
    train_task = train_model(
        preprocessed_dataset=preprocess_task.outputs["output_dataset"]
    )

    # Step 4: Evaluation
    evaluate_task = evaluate_model(
        model=train_task.outputs["model"],
        preprocessed_dataset=preprocess_task.outputs["output_dataset"]
    )