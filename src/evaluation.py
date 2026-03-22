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
        nn.Linear(checkpoint["hidden_size"], checkpoint["hidden_size"] // 2),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_size"] // 2, 1)
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
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f9;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                }}
                .report {{
                    background-color: #ffffff;
                    padding: 30px 50px;
                    border-radius: 15px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    text-align: center;
                }}
                h1 {{
                    color: #2c3e50;
                    margin-bottom: 20px;
                }}
                p {{
                    font-size: 18px;
                    margin: 10px 0;
                }}
                .mse {{ color: #e74c3c; font-weight: bold; }}
                .r2 {{ color: #27ae60; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="report">
                <h1>Evaluation Report</h1>
                <p class="mse">MSE: {mse:.4f}</p>
                <p class="r2">R²: {r2:.4f}</p>
            </div>
        </body>
        </html>
        """)
    print(f"Evaluation complete. Metrics logged and report saved to {html.path}")