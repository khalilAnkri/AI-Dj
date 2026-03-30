FROM python:3.13-slim

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml .
 
RUN uv pip install --system --no-cache -e .
 
COPY . .

EXPOSE 8080

# Put exact location of app.py
CMD ["uvicorn", "src.ui.app:app", "--host", "0.0.0.0", "--port", "8080"]