FROM python:3.13-slim

WORKDIR /src

RUN pip install --no-cache-dir flask

COPY src/ ./src/
COPY utils/ ./utils/

EXPOSE 8080

ENV FLASK_APP=src/api/main.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]