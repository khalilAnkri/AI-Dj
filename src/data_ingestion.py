from kfp.dsl import component, Output, Dataset

BASE_IMAGE = "europe-west1-docker.pkg.dev/ai-dj-487610/vertex-ai-pipeline-example/pipeline-base:latest"

@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "pandas",
        "pyarrow",
        "scikit-learn",
        "torch",
        "google-cloud-bigquery"
    ]
)
def data_ingestion(
    bq_project: str,
    bq_dataset: str,
    bq_table: str,
    dataset: Output[Dataset]
):
    """
    Extract a BigQuery table and save it as a single Parquet file to the pipeline output folder (GCS URI required).
    """

    from google.cloud import bigquery
    import os

    client = bigquery.Client(project=bq_project)


    destination_uri = f"{dataset.uri}/data.parquet" 

    extract_job = client.extract_table(
        f"{bq_project}.{bq_dataset}.{bq_table}",
        destination_uris=[destination_uri],
        job_config=bigquery.ExtractJobConfig(
            destination_format=bigquery.DestinationFormat.PARQUET
        )
    )

    extract_job.result()
    print(f"✅ Exported BigQuery table {bq_dataset}.{bq_table} to {destination_uri}")