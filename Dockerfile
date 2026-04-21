FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    pydantic>=2.0.0 \
    uvicorn>=0.24.0 \
    requests>=2.31.0 \
    openai>=1.0.0 \
    rich>=13.0.0 \
    matplotlib>=3.8.0

# Set PYTHONPATH so all imports work
ENV PYTHONPATH="/app"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Serve the Multi-Agent War Room environment (Round 2)
CMD ["uvicorn", "round2.war_room.app:app", "--host", "0.0.0.0", "--port", "8000"]
