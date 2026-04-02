"""FastAPI application for the SRE Incident Response environment."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
import uvicorn

from sre_env.server.sre_environment import SREEnvironment
from sre_env.server.models import SREAction

app = FastAPI(
    title="SRE Incident Response Environment",
    description="OpenEnv-compliant SRE incident response simulation",
    version="0.1.0",
)

# Global environment instance (per-session in production)
env = SREEnvironment()


class ResetRequest(BaseModel):
    task_id: str = "task1"
    seed: Optional[int] = 42
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    command: str


@app.get("/")
async def root():
    from fastapi.responses import HTMLResponse
    return HTMLResponse("""
    <html><head><title>SRE Incident Response Environment</title></head>
    <body style="font-family:monospace;max-width:700px;margin:40px auto;padding:0 20px">
    <h1>🔧 SRE Incident Response Environment</h1>
    <p>OpenEnv-compliant SRE incident response simulation.</p>
    <h3>API Endpoints</h3>
    <ul>
    <li><b>GET</b> <a href="/health">/health</a> — Health check</li>
    <li><b>POST</b> /reset — Reset environment (body: {"task_id":"task1","seed":42})</li>
    <li><b>POST</b> /step — Execute command (body: {"command":"ps aux"})</li>
    <li><b>GET</b> <a href="/state">/state</a> — Current state</li>
    <li><b>GET</b> <a href="/schema">/schema</a> — Action/observation schema</li>
    <li><b>GET</b> <a href="/docs">/docs</a> — Interactive API docs</li>
    </ul>
    <h3>Tasks</h3>
    <ul>
    <li><b>task1</b> (Easy) — Service Restart: nginx has crashed</li>
    <li><b>task2</b> (Medium) — Memory Leak Diagnosis: OOM-killing process</li>
    <li><b>task3</b> (Hard) — Cascading Failure: wrong DB credentials cascade</li>
    </ul>
    </body></html>
    """)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(req: ResetRequest):
    obs = env.reset(task_id=req.task_id, seed=req.seed, episode_id=req.episode_id)
    return obs.model_dump()


@app.post("/step")
async def step(req: StepRequest):
    obs = env.step(SREAction(command=req.command))
    return obs.model_dump()


@app.get("/state")
async def state():
    return env.state.model_dump()


@app.get("/schema")
async def schema():
    return {
        "action": SREAction.model_json_schema(),
        "observation": {"type": "object", "description": "SRE observation with terminal output"},
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_env = SREEnvironment()
    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type", "")

            if msg_type == "reset":
                task_id = data.get("data", {}).get("task_id", "task1")
                seed = data.get("data", {}).get("seed", 42)
                obs = session_env.reset(task_id=task_id, seed=seed)
                await ws.send_json({"type": "observation", "data": obs.model_dump()})

            elif msg_type == "step":
                command = data.get("data", {}).get("command", "help")
                obs = session_env.step(SREAction(command=command))
                await ws.send_json({"type": "observation", "data": obs.model_dump()})

            elif msg_type == "state":
                await ws.send_json({"type": "state", "data": session_env.state.model_dump()})

            elif msg_type == "close":
                await ws.close()
                break

            else:
                await ws.send_json({"type": "error", "data": {"message": f"Unknown type: {msg_type}"}})

    except WebSocketDisconnect:
        pass


def main():
    uvicorn.run("sre_env.server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
