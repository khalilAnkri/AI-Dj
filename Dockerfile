FROM python:3.13-slim

WORKDIR /app

# install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files and install them
COPY pyproject.toml uv.lock .
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY hello.py .

# Add the virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8080

CMD ["flask", "--app", "hello", "run", "--host=0.0.0.0", "--port=8080"]

# Build the image
# docker build -t my-flask-app .

# Run a container, mapping host port 9090 to container port 8080
# docker run -p 9090:8080 my-flask-app