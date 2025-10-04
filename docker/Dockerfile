# syntax=docker/dockerfile:1.5
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_CONFIG_PATH=/app/config/pipeline.yaml

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["uvicorn", "src.api.server:build_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
