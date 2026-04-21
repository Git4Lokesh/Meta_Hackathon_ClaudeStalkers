FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir fastapi pydantic uvicorn requests openai

ENV PYTHONPATH="/app"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "round2.war_room.app:app", "--host", "0.0.0.0", "--port", "8000"]
