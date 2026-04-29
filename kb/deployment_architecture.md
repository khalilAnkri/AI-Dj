# Deployment architecture

## Serving

The trained KNN model is wrapped behind a FastAPI application defined in
`src/api/main.py`. The image is built with a `python:3.12-slim` base, uses
`uv pip install --system` for dependencies, and runs with `uvicorn` on the
port provided by the `PORT` environment variable.

## Hosting

The container image is pushed to Google Artifact Registry and deployed on
Cloud Run in `europe-west1`. Cloud Run scales the service to zero when idle,
which keeps cost low for a teaching project. Outbound calls to Vertex AI
(used by the Lab 9 agent) are authenticated via the Cloud Run service
account, so no API key is shipped in the image.

## Data and feature flow

Track audio features are pulled from Spotify (via Spotipy) and from the
project's BigQuery dataset. The training pipeline is orchestrated with KFP
on Vertex AI Pipelines (`ai-dj-training-pipeline.json`). Artifacts and
trained models are persisted in Google Cloud Storage.

## Monitoring and experimentation

- Experiment tracking is documented in `docs/Experimentation.md`.
- The training pipeline definition lives in `src/pipelines/`.
- The serving layer lives in `src/api/`.
- The Lab 9 agent layer lives in `src/team.py`, `src/first_agent.py` and
  `src/app.py`, with knowledge base documents in `kb/`.

## Stack summary

- Python 3.12 managed with `uv`.
- FastAPI + Uvicorn for the prediction API.
- Flask + Gunicorn for the agent API (Lab 9).
- ChromaDB (in-memory) as the vector store for the agent's knowledge base.
- Vertex AI Gemini 2.5 Flash as the LLM behind the agent.
- Cloud Run for deployment, Artifact Registry for images, BigQuery for data.
