from kfp.dsl import Dataset, Input, Metrics, Model, Output, component

from src.pipelines.config import BASE_IMAGE


@component(
    base_image=BASE_IMAGE, packages_to_install=["pandas", "scikit-learn", "pyarrow"]
)
def train_model(
    preprocessed_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100,
):
    import pickle

    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # 1. Load the TRAINING split (80% of data)
    df = pd.read_parquet(preprocessed_dataset.path)

    # 2. Split features and target
    X = df.drop(columns=["hit"])
    y = df["hit"]

    # 3. Train model
    model_instance = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
    )

    model_instance.fit(X, y)

    # 4. Internal Training Check
    y_pred = model_instance.predict(X)
    train_acc = accuracy_score(y, y_pred)

    # Log for Experiment Tracking
    metrics.log_metric("training_accuracy", float(train_acc))
    metrics.log_metric("n_estimators", n_estimators)

    # 5. Feature importance
    importance = model_instance.feature_importances_
    feature_imp = pd.DataFrame({"feature": X.columns, "importance": importance})
    top_5 = (
        feature_imp.sort_values(by="importance", ascending=False)["feature"]
        .head(5)
        .tolist()
    )

    # 6. Save outputs as .pkl
    model_file_path = model.path + ".pkl"
    with open(model_file_path, "wb") as f:
        pickle.dump(model_instance, f)

    # Save metadata for the Model Registry
    model.metadata["framework"] = "scikit-learn"
    model.metadata["top_5_features"] = top_5
    model.metadata["columns"] = X.columns.tolist()

    print(f"Model saved. Training Accuracy: {train_acc}")
