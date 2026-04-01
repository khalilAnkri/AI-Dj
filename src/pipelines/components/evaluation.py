from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

from src.pipelines.config import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=[
        "pandas",
        "scikit-learn",
        "pyarrow"
    ]
)
def evaluate_model(
    model: Input[Model],
    preprocessed_dataset: Input[Dataset],
    metrics: Output[Metrics],
):
    import pickle

    import pandas as pd
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score)

    # 1. Load the TEST data
    df = pd.read_parquet(preprocessed_dataset.path)
    X = df.drop(columns=["hit"])
    y = df["hit"]

    # 2. Load the EXACT .pkl model produced by the training component
    model_file_path = model.path + ".pkl"
    with open(model_file_path, "rb") as f:
        model_instance = pickle.load(f)

    # 3. Predict on the test dataset
    y_pred = model_instance.predict(X)

    # 4. Calculate Detailed Metrics
    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred))
    f1 = float(f1_score(y, y_pred))

    # 5. Log to Vertex AI Metadata (The MLOps Way)
    metrics.log_metric("test_accuracy", acc)
    metrics.log_metric("test_precision", prec)
    metrics.log_metric("test_recall", rec)
    metrics.log_metric("test_f1_score", f1)

    print(f"Evaluation complete. Test Accuracy: {acc}")
