FROM mirror.gcr.io/library/python:3.11-slim

WORKDIR /app

# Set the PYTHONPATH here so it's always available
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything from src into /app/src
COPY src/ ./src/

# Now you can call api.main directly because /app/src is in the path
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]