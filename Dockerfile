# Use 3.12 for maximum stability with Vertex AI libraries
FROM python:3.12-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files first to cache the 'install' layer
COPY pyproject.toml .
# If you have a requirements.txt instead, use: COPY requirements.txt .

# Install dependencies using uv into the system python
RUN uv pip install --system --no-cache fastapi uvicorn google-cloud-aiplatform spotipy python-dotenv pandas

# Copy the rest of the source code
COPY . .

# Cloud Run uses the PORT env variable
EXPOSE 8080

# The CMD needs to point to the file where 'app = FastAPI()' lives
# use the API path we built :
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]