from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
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
def train_model(
    preprocessed_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics], 
    n_estimators: int = 100, 
):
    import pandas as pd
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # 1. Load data using the direct path
    df = pd.read_parquet(preprocessed_dataset.path)

    # 2. Split features and target
    X = df.drop(columns=["hit"])
    y = df["hit"]

    # 3. Train model
    model_instance = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )

    model_instance.fit(X, y)

    # 4. Evaluation (Basic internal check)
    y_pred = model_instance.predict(X)
    acc = accuracy_score(y, y_pred)
    
    # Log to Vertex AI Metrics artifact
    metrics.log_metric("accuracy", float(acc))
    metrics.log_metric("n_estimators", n_estimators)

    # 5. Feature importance
    importance = model_instance.feature_importances_
    feature_imp = pd.DataFrame({"feature": X.columns, "importance": importance})
    top_5 = feature_imp.sort_values(by="importance", ascending=False)["feature"].head(5).tolist()

    # 6. Save outputs
    suffix = ".joblib"
    joblib.dump(model_instance, model.path + suffix)
    
    # Save metadata to the Model artifact
    model.metadata["framework"] = "scikit-learn"
    model.metadata["top_5_features"] = top_5
    model.metadata["columns"] = X.columns.tolist()

    print(f"Model saved with accuracy: {acc}")