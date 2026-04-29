FROM mirror.gcr.io/library/python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN uv export --frozen --no-dev --no-hashes -o /tmp/requirements.txt && \
    uv pip install --system -r /tmp/requirements.txt

COPY src/ ./src/

EXPOSE 8501

ENV PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true


# ENV GOOGLE_APPLICATION_CREDENTIALS="/utils/ai-dj-487610-8ac3df85f59d.json"

CMD streamlit run src/data_visualization.py --server.port $PORT --server.address=0.0.0.0