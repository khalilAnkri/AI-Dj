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
        "google-cloud-bigquery",
        "joblib"
    ]
)
def preprocessing(
    input_dataset: Input[Dataset],
    preprocessed_dataset: Output[Dataset],
):
    import os
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # -----------------------------
    # 1. Read dataset
    # -----------------------------
    df = pd.read_parquet(os.path.join(input_dataset.path, "data.parquet"))
    target = "price"

    # -----------------------------
    # 2. Identify numeric & categorical
    # -----------------------------
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target in numerical_cols:
        numerical_cols.remove(target)

    # -----------------------------
    # 3. Handle missing values
    # -----------------------------
    df[numerical_cols] = df[numerical_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    # -----------------------------
    # 4. Scale numerical features
    # -----------------------------
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # -----------------------------
    # 5. Encode categorical features
    # -----------------------------
    for col in categorical_cols:
        counts = df[col].value_counts()
        rare = counts[counts < 10].index
        df[col] = df[col].replace(rare, "Other")

    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[categorical_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    else:
        encoded_df = pd.DataFrame()

    # -----------------------------
    # 6. Normalize target
    # -----------------------------
    target_scaler = StandardScaler()
    df[target] = target_scaler.fit_transform(df[[target]])

    # -----------------------------
    # 7. Combine all features
    # -----------------------------
    df_processed = pd.concat([df[numerical_cols], encoded_df, df[target]], axis=1)

    # -----------------------------
    # 8. Save preprocessed data
    # -----------------------------
    os.makedirs(preprocessed_dataset.path, exist_ok=True)
    df_processed.to_parquet(os.path.join(preprocessed_dataset.path, "data.parquet"))

    # -----------------------------
    # 9. Save scalers & encoders
    # -----------------------------
    joblib.dump(scaler, os.path.join(preprocessed_dataset.path, "feature_scaler.pkl"))
    if len(categorical_cols) > 0:
        joblib.dump(encoder, os.path.join(preprocessed_dataset.path, "onehot_encoder.pkl"))
    joblib.dump(target_scaler, os.path.join(preprocessed_dataset.path, "target_scaler.pkl"))

    print(f"Preprocessing complete. Output saved to {preprocessed_dataset.path}")