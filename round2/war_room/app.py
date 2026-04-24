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
    <body style="font-family:monospace;max-width:700px;margin:40px auto;padding:0 20px;background:#0d1117;color:#c9d1d9">
    <h1>🔥 Multi-Agent Incident War Room</h1>
    <p><b>Team ClaudeStalkers</b> — BITS Pilani Hyderabad<br>
    Meta PyTorch OpenEnv Hackathon 2026</p>
    <p>Three AI agents cooperate to diagnose and fix production incidents under partial observability, phantom alerts, and adversarial noise.</p>

    <h3>Agents (role-based partial observability)</h3>
    <ul>
    <li>🚨 <b>Triage</b> — Dashboard, alerts, health metrics</li>
    <li>🔎 <b>Diagnosis</b> — Log files, process table</li>
    <li>🛠️ <b>Remediation</b> — Service status, config files, restart commands</li>
    </ul>

    <h3>API Endpoints</h3>
    <ul>
    <li><b>GET</b> <a href="/health" style="color:#58a6ff">/health</a></li>
    <li><b>POST</b> /reset — {task_id, seed}</li>
    <li><b>POST</b> /step — MultiAgentAction</li>
    <li><b>GET</b> <a href="/state" style="color:#58a6ff">/state</a></li>
    <li><b>GET</b> <a href="/schema" style="color:#58a6ff">/schema</a> — JSON schemas</li>
    <li><b>GET</b> <a href="/docs" style="color:#58a6ff">/docs</a> — Interactive Swagger docs</li>
    </ul>

    <h3>6 Escalating Tasks</h3>
    <ul>
    <li><b>task1</b> (Easy, 10 rounds) — Coordinated nginx restart</li>
    <li><b>task2</b> (Medium, 15 rounds) — Memory leak + CPU red herring</li>
    <li><b>task3</b> (Hard, 20 rounds) — Cascading DB failure + phantom Redis alerts (Theory of Mind)</li>
    <li><b>task4</b> (Expert, 25 rounds) — Simultaneous incidents</li>
    <li><b>task5</b> (Expert, 20 rounds) — Rogue insider threat</li>
    <li><b>task6</b> (Expert, 25 rounds) — Blame game with conflicting reports</li>
    </ul>

    <h3>Links</h3>
    <ul>
    <li><a href="https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers" style="color:#58a6ff">GitHub Repo</a></li>
    <li><a href="https://huggingface.co/spaces/brodie1of1/war-room" style="color:#58a6ff">HF Spaces</a></li>
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
    uvicorn.run("round2.war_room.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
