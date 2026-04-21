"""FastAPI server for the Multi-Agent Incident War Room."""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import MultiAgentAction, AgentAction, Message

app = FastAPI(
    title="Multi-Agent Incident War Room",
    description="OpenEnv-compliant multi-agent SRE incident response environment",
    version="0.1.0",
)

env = WarRoomEnvironment()


class ResetRequest(BaseModel):
    task_id: str = "task1"
    seed: Optional[int] = 42


@app.get("/")
async def root():
    return HTMLResponse("""
    <html><head><title>Multi-Agent Incident War Room</title></head>
    <body style="font-family:monospace;max-width:700px;margin:40px auto;padding:0 20px">
    <h1>🔧 Multi-Agent Incident War Room</h1>
    <p>Three AI agents cooperate to diagnose and fix production incidents.</p>
    <h3>Agents</h3>
    <ul>
    <li><b>Triage</b> — Monitors dashboard, escalates issues</li>
    <li><b>Diagnosis</b> — Reads logs, identifies root causes</li>
    <li><b>Remediation</b> — Fixes configs, restarts services</li>
    </ul>
    <h3>API</h3>
    <ul>
    <li><b>GET</b> <a href="/health">/health</a></li>
    <li><b>POST</b> /reset — {task_id, seed}</li>
    <li><b>POST</b> /step — MultiAgentAction</li>
    <li><b>GET</b> <a href="/state">/state</a></li>
    <li><b>GET</b> <a href="/docs">/docs</a> — Interactive docs</li>
    </ul>
    <h3>Tasks</h3>
    <ul>
    <li><b>task1</b> (Easy) — Coordinated Service Restart</li>
    <li><b>task2</b> (Medium) — Memory Leak with Misdirection</li>
    <li><b>task3</b> (Hard) — Cascading Failure with Conflicting Info</li>
    <li><b>task4</b> (Expert) — Simultaneous Incidents</li>
    </ul>
    </body></html>
    """)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(req: Optional[ResetRequest] = None):
    if req is None:
        req = ResetRequest()
    obs = env.reset(task_id=req.task_id, seed=req.seed)
    return obs.model_dump()


@app.post("/step")
async def step(action: MultiAgentAction):
    obs = env.step(action)
    return obs.model_dump()


@app.get("/state")
async def state():
    return env.state.model_dump()


@app.get("/schema")
async def schema():
    return {
        "action": MultiAgentAction.model_json_schema(),
        "observation": {"type": "object", "description": "Multi-agent observation"},
    }


def main():
    uvicorn.run("round2.war_room.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
