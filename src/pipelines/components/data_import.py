from kfp.dsl import Dataset, Output, component

from src.pipelines.config import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "pyarrow", "google-cloud-bigquery", "db-dtypes"],
)
def import_bigquery(
    output_dataset: Output[Dataset],
    project_id: str,
):
    from google.cloud import bigquery

    # 1. Initialize client with explicit project
    client = bigquery.Client(project=project_id)

    query = """
    SELECT *
    FROM `ai-dj-487610.Spotify_Tracks.tracks`
    """

    # 2. Add progress logging to see where it hangs in the Vertex logs
    print("Starting BigQuery Query...")

    df = client.query(query).to_dataframe()

    print(f"Query successful. Retrieved {len(df)} rows.")

    # 3. Explicitly use pyarrow engine and ensure path is handled
    df.to_parquet(output_dataset.path, engine="pyarrow", index=False)

    # 4. Metadata for the Vertex UI
    output_dataset.metadata["row_count"] = len(df)
    output_dataset.metadata["source"] = "BigQuery"

    print(f"Dataset saved to {output_dataset.path}")
