FROM python:3.13-slim

WORKDIR /src

COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

CMD ["python", "tests/train-mpirlet.py"]

# Build: 
# docker build -t ml-workflow-container .
# 
# Run it: 
# docker run ml-workflow-container