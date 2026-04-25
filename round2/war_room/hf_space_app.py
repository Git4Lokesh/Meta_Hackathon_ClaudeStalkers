"""Hugging Face Spaces entry point.

Mounts both surfaces on a single process:
  * Gradio UI at  /              (interactive demo for judges)
  * OpenEnv API  /api/reset      (reset episode)
                 /api/step       (step the environment)
                 /api/state      (current state)
                 /api/health     (liveness check)
                 /api/schema     (JSON schemas)

Usage (local):
    HF_TOKEN=... PYTHONPATH=. python round2/war_room/hf_space_app.py

Usage (Docker/HF Spaces):
    uvicorn round2.war_room.hf_space_app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# Reuse the OpenEnv REST endpoints defined in app.py. We wrap that app
# under /api so the Gradio UI can own "/".
from round2.war_room.app import app as openenv_app
from round2.war_room.gradio_app import build_app as build_gradio_app


def create_app() -> FastAPI:
    """Compose the Gradio demo with the OpenEnv REST API.

    Returns a FastAPI app ready to run under uvicorn. Gradio is mounted
    at /, the OpenEnv endpoints are preserved under /api/*.
    """
    # Top-level FastAPI shell — Gradio will be mounted on it directly.
    app = FastAPI(
        title="Multi-Agent Incident War Room",
        description=(
            "Multi-agent SRE incident environment. Gradio UI at /, "
            "OpenEnv REST API at /api/*."
        ),
        version="0.2.0",
    )

    # Mount OpenEnv REST surface under /api so it stays a first-class
    # citizen for benchmarking / programmatic use.
    app.mount("/api", openenv_app)

    # Top-level convenience redirects so judges don't 404 on old URLs.
    @app.get("/health")
    async def _health_alias():  # pragma: no cover - trivial alias
        return RedirectResponse(url="/api/health")

    # Build the Gradio Blocks and mount at / so visiting the Space
    # lands straight on the interactive dashboard.
    gradio_blocks = build_gradio_app()
    app = gr.mount_gradio_app(app, gradio_blocks, path="/")

    return app


# Module-level `app` so `uvicorn round2.war_room.hf_space_app:app` works.
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "round2.war_room.hf_space_app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )
