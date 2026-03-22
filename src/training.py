from kfp.dsl import component, Input, Output, Dataset, Model, Metrics
import os

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
    from sklearn.preprocessing import StandardScaler

    # Load data
    df = pd.read_parquet(os.path.join(preprocessed_dataset.path, "data.parquet"))
    df = df.fillna(0)  # Handle missing values

    target = "price"

    X = df.drop(columns=[target]).values

    # Normalize target
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(df[target].values.reshape(-1, 1))

    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_scaled, test_size=0.2, random_state=42
    )

    class HouseDataset(TorchDataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_loader = DataLoader(
        HouseDataset(X_train, y_train),
        batch_size=hyperparameters.get("batch_size", 32),
        shuffle=True
    )
    val_loader = DataLoader(
        HouseDataset(X_val, y_val),
        batch_size=hyperparameters.get("batch_size", 32)
    )

    # Define model with 2 hidden layers
    input_size = X.shape[1]
    hidden_size = hyperparameters.get("hidden_size", 64)

    model_nn = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Linear(hidden_size // 2, 1)
    )

    optimizer = torch.optim.Adam(model_nn.parameters(), lr=hyperparameters.get("lr", 1e-3))
    loss_fn = nn.MSELoss()

    # Training loop
    epochs = hyperparameters.get("epochs", 50)
    for epoch in range(epochs):
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

    # Inverse scale target to compute metrics on real values
    preds_real = target_scaler.inverse_transform(preds)
    truths_real = target_scaler.inverse_transform(truths)

    mse = mean_squared_error(truths_real, preds_real)
    r2 = r2_score(truths_real, preds_real)

    metrics.log_metric("mse", mse.item() if hasattr(mse, 'item') else float(mse))
    metrics.log_metric("r2", r2.item() if hasattr(r2, 'item') else float(r2))
    print(f"Training complete. MSE: {mse:.2f}, R2: {r2:.4f}")

    # Save model
    os.makedirs(model.path, exist_ok=True)
    torch.save({
        "state_dict": model_nn.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "target_scaler_mean": target_scaler.mean_[0].item() if hasattr(target_scaler.mean_[0], 'item') else float(target_scaler.mean_[0]),
        "target_scaler_scale": target_scaler.scale_[0].item() if hasattr(target_scaler.scale_[0], 'item') else float(target_scaler.scale_[0])
    }, os.path.join(model.path, "model.pth"))

    print(f"Model saved to {os.path.join(model.path, 'model.pth')}")