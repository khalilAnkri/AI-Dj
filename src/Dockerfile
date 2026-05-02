FROM mirror.gcr.io/library/python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN uv export --frozen --no-dev --no-hashes -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt

COPY src/ ./src/
COPY data/ ./data/

ENV PORT=8080

# the argument timetout 120 is important to not kill the request
CMD ["gunicorn", "--bind", ":8080", "--workers", "1", "--threads", "4", "--timeout", "120", "src.app:app"]