from google.cloud import aiplatform
from kfp import compiler

from src.pipelines.config import BUCKET_NAME, PIPELINE_NAME, PROJECT_ID, REGION
from src.pipelines.pipeline import ai_dj_training_workflow as pipeline


def main():
    output_file = f"{PIPELINE_NAME}.json"

    # 1. Compile the pipeline
    compiler.Compiler().compile(pipeline_func=pipeline, package_path=output_file)
    print(f"Pipeline compiled successfully → {output_file}")

    # 2. Initialize Vertex AI SDK
    aiplatform.init(
        project=PROJECT_ID,
        location=REGION,
        staging_bucket=f"gs://{BUCKET_NAME}/staging",
    )

    # 3. Create the Pipeline Job
    job = aiplatform.PipelineJob(
        display_name=f"{PIPELINE_NAME}-run",
        template_path=output_file,
        pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root",
        enable_caching=True,  # Recommended to save time/cost during debugging
    )

    # 4. Submit the job
    job.submit(service_account="343602206157-compute@developer.gserviceaccount.com")
    print("Job submitted to Vertex AI. Check the console!")


if __name__ == "__main__":
    main()
