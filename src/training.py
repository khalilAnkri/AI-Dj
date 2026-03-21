from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

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
def training(
    preprocessed_dataset: Input[Dataset],
    model: Output[Model],
    metrics: Output[Metrics],
    hyperparameters: dict
):
    import os
    import pandas as pd
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # Load data
    df = pd.read_parquet(os.path.join(preprocessed_dataset.path, "data.parquet"))
    target = "price"

    X = df.drop(columns=[target]).values
    y = df[target].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    class HouseDataset(TorchDataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_loader = DataLoader(HouseDataset(X_train, y_train), batch_size=hyperparameters["batch_size"], shuffle=True)
    val_loader = DataLoader(HouseDataset(X_val, y_val), batch_size=hyperparameters["batch_size"])

    # Define model
    input_size = X.shape[1]
    hidden_size = hyperparameters["hidden_size"]

    model_nn = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
    )

    optimizer = torch.optim.Adam(model_nn.parameters(), lr=hyperparameters["lr"])
    loss_fn = nn.MSELoss()

    # Training loop
    for epoch in range(hyperparameters["epochs"]):
        model_nn.train()
        for xb, yb in train_loader:
            pred = model_nn(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation
    model_nn.eval()
    preds, truths = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            p = model_nn(xb)
            preds.extend(p.cpu().numpy())
            truths.extend(yb.cpu().numpy())

    # Log metrics
    mse = mean_squared_error(truths, preds)
    r2 = r2_score(truths, preds)
    metrics.log_metric("mse", float(mse))
    metrics.log_metric("r2", float(r2))

    # Save model
    os.makedirs(model.path, exist_ok=True)
    torch.save({
        "state_dict": model_nn.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size
    }, os.path.join(model.path, "model.pth"))

    print(f"Training complete. Model saved to {os.path.join(model.path, 'model.pth')}")