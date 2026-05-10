from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_read_main():
    """Verify the API is up and running"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_endpoint_valid_data():
    """
    Test the core prediction logic with valid input
    Ensures the model loader and FastAPI work together
    """
    valid_payload = {
        "danceability": 0.7,
        "energy": 0.8,
        "loudness": -5.0,
        "tempo": 120.0,
    }
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], (int, float))


def test_prediction_invalid_data():
    """
    Validate input schema
    """
    invalid_payload = {"danceability": "very_high"}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
