from kfp import dsl, compiler
from google.cloud import aiplatform

# Import your components
from src.data_ingestion import data_ingestion
from src.preprocessing import preprocessing
from src.training import training
from src.evaluation import evaluation
from src.config import BASE_IMAGE

# =========================
# CONFIG
# =========================
PROJECT_ID    = "ai-dj-487610"
BUCKET_NAME   = "gs://mlops-labs-bucket"
BQ_DATASET    = "housing_dataset"
BQ_TABLE      = "housing"
LOCATION      = "europe-west1"

PIPELINE_ROOT = f"{BUCKET_NAME}/pipeline_root_houseprice/"

# =========================
# PIPELINE DEFINITION
# =========================
@dsl.pipeline(
    name="houseprice-pipeline",
    pipeline_root=PIPELINE_ROOT
)
def houseprice_pipeline():

    # 1️⃣ Data ingestion
    ingestion_task = data_ingestion(
        bq_project=PROJECT_ID,
        bq_dataset=BQ_DATASET,
        bq_table=BQ_TABLE
    )

    # 2️⃣ Preprocessing
    preprocessing_task = preprocessing(
        input_dataset=ingestion_task.outputs["dataset"]
    )

    # 3️⃣ Training
    training_task = training(
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"],
        hyperparameters={
            "lr": 0.001,
            "epochs": 50,
            "batch_size": 32,
            "hidden_size": 64
        }
    )

    # 4️⃣ Evaluation
    evaluation_task = evaluation(
        model=training_task.outputs["model"],
        preprocessed_dataset=preprocessing_task.outputs["preprocessed_dataset"]
    )

# =========================
# COMPILE + RUN
# =========================
if __name__ == "__main__":

    # Compile pipeline
    compiler.Compiler().compile(
        pipeline_func=houseprice_pipeline,
        package_path="houseprice_pipeline.json"
    )

    # Initialize Vertex AI
    aiplatform.init(
        project=PROJECT_ID,
        location=LOCATION,
        staging_bucket=BUCKET_NAME
    )

    # Submit pipeline job
    job = aiplatform.PipelineJob(
        display_name="houseprice-pipeline-run",
        template_path="houseprice_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
    )

    job.run(sync=True)