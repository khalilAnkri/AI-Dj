from kfp.dsl import component, Input, Output, Dataset, Model, Metrics, HTML

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
def evaluation(
    model: Input[Model],
    preprocessed_dataset: Input[Dataset],
    metrics: Output[Metrics],
    html: Output[HTML]
):
    import os
    import pandas as pd
    import torch
    import torch.nn as nn
    from sklearn.metrics import mean_squared_error, r2_score

    # Load preprocessed data
    df = pd.read_parquet(os.path.join(preprocessed_dataset.path, "data.parquet"))
    target = "price"
    X = df.drop(columns=[target]).values
    y = df[target].values

    # Load trained model
    checkpoint = torch.load(os.path.join(model.path, "model.pth"))

    model_nn = nn.Sequential(
        nn.Linear(checkpoint["input_size"], checkpoint["hidden_size"]),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_size"], 1)
    )
    model_nn.load_state_dict(checkpoint["state_dict"])
    model_nn.eval()

    # Predictions
    with torch.no_grad():
        preds = model_nn(torch.tensor(X, dtype=torch.float32)).cpu().numpy()

    # Metrics
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    metrics.log_metric("mse", float(mse))
    metrics.log_metric("r2", float(r2))

    # Save HTML report
    os.makedirs(os.path.dirname(html.path), exist_ok=True)
    with open(html.path, "w") as f:
        f.write(f"""
        <html>
        <body>
        <h1>Evaluation Report</h1>
        <p>MSE: {mse:.4f}</p>
        <p>R2: {r2:.4f}</p>
        </body>
        </html>
        """)

    print(f"Evaluation complete. Metrics logged and report saved to {html.path}")