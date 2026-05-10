from kfp.dsl import Dataset, Input, Output, component

from src.pipelines.config import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "pandas",
        "pyarrow",
        "scikit-learn",
    ],  # Add scikit-learn for splitting
)
def training_preprocess(
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],  # New: Specifically for Training
    test_dataset: Output[Dataset],  # New: Specifically for Evaluation
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # 1. Load data
    df = pd.read_parquet(input_dataset.path)

    # 2. Target Engineering
    df["hit"] = (df["popularity"] >= 65).astype(int)

    # 3. Feature Selection
    # Dropping non-feature columns and keeping only numeric
    df = df.drop(columns=["popularity", "Unnamed: 0"], errors="ignore")
    numeric_df = df.select_dtypes(include=["number"]).copy()

    # Ensure "hit" is included in the final set before splitting
    if "hit" not in numeric_df.columns:
        numeric_df["hit"] = df["hit"]

    # 4. THE FIX: The Train/Test Split
    # We set aside 20% of the data that the model will NEVER see during training
    train_df, test_df = train_test_split(
        numeric_df,
        test_size=0.2,
        random_state=42,
        stratify=numeric_df["hit"],  # Ensures equal hit/miss ratio in both sets
    )

    # 5. Save separate artifacts
    train_df.to_parquet(train_dataset.path)
    test_df.to_parquet(test_dataset.path)

    print("Preprocessing complete.")
    print(f"Training set: {len(train_df)} rows | Test set: {len(test_df)} rows")
