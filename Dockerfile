FROM python:3.13-slim

WORKDIR /app

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies into the .venv
# Note: Ensure you ran 'uv add gunicorn numpy scikit-learn' locally first!
RUN uv sync --frozen --no-dev --no-install-project

# FIX: Copy the application code, the model, and the templates folder
COPY hello.py .
COPY model.pkl .
COPY templates/ ./templates/

# Make venv binaries available (this is where gunicorn lives)
ENV PATH="/app/.venv/bin:$PATH"

# Cloud Run requirement
EXPOSE 8080

# Production command
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "8", "hello:app"]