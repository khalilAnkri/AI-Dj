from kfp.dsl import component, Input, Output, Dataset
import os

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
def preprocessing(
    input_dataset: Input[Dataset],
    preprocessed_dataset: Output[Dataset],
):
    import os   
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # Read parquet file
    df = pd.read_parquet(os.path.join(input_dataset.path, "data.parquet"))

    target = "price"

    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if target in numerical_cols:
        numerical_cols.remove(target)

    # Scale numerical
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Encode categorical
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    else:
        encoded_df = pd.DataFrame()

    df_processed = pd.concat(
        [df[numerical_cols], encoded_df, df[target]],
        axis=1
    )

    os.makedirs(preprocessed_dataset.path, exist_ok=True)
    df_processed.to_parquet(os.path.join(preprocessed_dataset.path, "data.parquet"))

    print(f"Preprocessing complete. Output saved to {preprocessed_dataset.path}")