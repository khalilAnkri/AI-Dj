from kfp.dsl import component, Input, Output, Dataset
from src.pipelines.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE,
    packages_to_install=["pandas", "pyarrow"]
)
def training_preprocess(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset],
):
    import pandas as pd
    

    # input_dataset.path is the exact location of the parquet file from the previous step.
    df = pd.read_parquet(input_dataset.path)

    # Creating the target variable for the AI DJ
    df["hit"] = (df["popularity"] >= 65).astype(int)
    
    # Dropping non-feature columns
    df = df.drop(columns=["popularity", "Unnamed: 0"], errors="ignore")

    # Feature Selection: Only numeric features for the models (RF/GB/KNN)
    X = df.drop("hit", axis=1)
    X = X.select_dtypes(include=["number"])
    y = df["hit"]

    # Combine into one dataframe for the training component
    processed_df = X.copy()
    processed_df["hit"] = y

    # Save output
    processed_df.to_parquet(output_dataset.path)

    print(f"Preprocessed dataset saved with {len(processed_df)} rows and {len(X.columns)} features.")