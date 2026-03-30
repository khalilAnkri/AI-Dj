FROM python:3.13-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
 
RUN uv pip install --system --no-cache -e .
 
COPY . .

EXPOSE 8080

# Put exact location of app.py; add variable port and parameters for docker
CMD ["sh", "-c", "uvicorn src.ui.app:app --host 0.0.0.0 --port ${PORT:-8080}"] 