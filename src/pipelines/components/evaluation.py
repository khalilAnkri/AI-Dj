from kfp.dsl import component, Input, Output, Model, Dataset, Metrics
from src.pipelines.config import BASE_IMAGE

@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "pyarrow",
        "joblib"
    ]
)
def evaluate_model(
    model: Input[Model],
    preprocessed_dataset: Input[Dataset],
    metrics: Output[Metrics], 
):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # 1. Load the data 
    df = pd.read_parquet(preprocessed_dataset.path)
    X = df.drop(columns=["hit"])
    y = df["hit"]

    # 2. Load the EXACT model produced by the training component
    model_instance = joblib.load(model.path + ".joblib")

    # 3. Predict on the dataset
    y_pred = model_instance.predict(X)

    # 4. Calculate Metrics
    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred))
    f1 = float(f1_score(y, y_pred))

    # 5. Log to Vertex AI Metadata (The MLOps Way)
    metrics.log_metric("accuracy", acc)
    metrics.log_metric("precision", prec)
    metrics.log_metric("recall", rec)
    metrics.log_metric("f1_score", f1)

    print(f"Evaluation complete. Accuracy: {acc}")