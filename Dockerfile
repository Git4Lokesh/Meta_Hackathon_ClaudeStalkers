FROM python:3.11-slim

WORKDIR /app

# System deps for curl (healthcheck) and basic building.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Project files
COPY . /app

# Python deps
# - OpenAI client is used by the Agent Mode rollout to call HF Inference
#   Providers; huggingface_hub is used for model/adapter metadata.
RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn>=0.24.0" \
    "requests>=2.31.0" \
    "openai>=1.0.0" \
    "huggingface_hub>=0.24.0" \
    "rich>=13.0.0" \
    "matplotlib>=3.8.0" \
    "gradio>=4.0.0"

ENV PYTHONPATH="/app"

# Health check targets the OpenEnv API so /health stays meaningful
# whether the Gradio UI is building or not.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

EXPOSE 7860

# Unified entry point:
#   Gradio UI at  /
#   OpenEnv API  /api/*
CMD ["uvicorn", "round2.war_room.hf_space_app:app", "--host", "0.0.0.0", "--port", "7860"]
