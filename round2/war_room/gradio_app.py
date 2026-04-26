"""Gradio Dashboard for the Multi-Agent Incident War Room.

Usage:
    pip install gradio matplotlib
    PYTHONPATH=. python3 round2/war_room/gradio_app.py
"""

import json
import os
import sys
from datetime import datetime
from typing import Optional

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import MultiAgentAction, AgentAction, Message
from round2.war_room.live_agent import LiveAgentConfig, LiveAgentRunner

# ---- Global state ----
env = WarRoomEnvironment()
current_obs = None
round_num = 0
chat_history = []
reward_history = []
milestone_list = []
thought_history = []
round_trace = []

# Live LLM runner (lazy init; only used when Agent Mode is enabled)
live_runner: Optional[LiveAgentRunner] = None
agent_mode_enabled: bool = False

ROLE_ICONS = {"triage": "🚨", "diagnosis": "🔎", "remediation": "🛠️"}
ROLE_COLORS = {"triage": "#FFD700", "diagnosis": "#00CED1", "remediation": "#32CD32"}

# ---- Task descriptions for the dropdown ----
TASK_DESCRIPTIONS = {
    "task1": "task1 — Coordinated Restart (🟢 Easy)",
    "task2": "task2 — Memory Leak + Red Herring (🟡 Medium)",
    "task3": "task3 — Cascading Failure + Phantom Alerts (🔴 Hard)",
    "task4": "task4 — Two Simultaneous Incidents (⚫ Expert)",
}

def _parse_task_key(task_id: str) -> str:
    """Extract the raw task key (e.g. 'task1') from a dropdown value or plain id."""
    # Try direct match first
    if task_id in HEURISTIC_STEPS:
        return task_id
    # Extract leading token before ' — '
    key = task_id.split(" ")[0] if " " in task_id else task_id
    if key in HEURISTIC_STEPS:
        return key
    # Fallback
    return "task1"

# ---- Custom CSS ----
CUSTOM_CSS = """
/* Dark theme overrides */
.gradio-container { background: #0a0a1a !important; }
.dark { background: #0a0a1a !important; }

/* Header banner */
.war-room-header {
    background: linear-gradient(135deg, #1a0000 0%, #0a0a2e 50%, #001a00 100%);
    border: 1px solid #333;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    text-align: center;
}
.war-room-header h1 { color: #ff4444; margin: 0; font-size: 2em; }
.war-room-header p { color: #888; margin: 4px 0 0 0; }

/* Chat panel */
.chat-container {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 16px;
    max-height: 360px;
    overflow-y: auto;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    margin-bottom: 12px;
}

/* Agent message */
.agent-msg {
    margin: 8px 0;
    padding: 12px 16px;
    border-radius: 8px;
    border-left: 4px solid;
    background: #161b22;
    animation: fadeIn 0.3s ease;
}
.agent-msg.triage { border-left-color: #FFD700; }
.agent-msg.diagnosis { border-left-color: #00CED1; }
.agent-msg.remediation { border-left-color: #32CD32; }

.agent-name { font-weight: 700; font-size: 0.9em; }
.agent-name.triage { color: #FFD700; }
.agent-name.diagnosis { color: #00CED1; }
.agent-name.remediation { color: #32CD32; }

.cmd-block {
    display: inline-block;
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 2px 8px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    color: #7ee787;
    font-size: 0.85em;
    margin: 4px 0;
}

.msg-bubble {
    color: #c9d1d9;
    font-style: italic;
    margin-top: 4px;
    padding-left: 8px;
    border-left: 2px solid #30363d;
}

.msg-target { color: #58a6ff; font-weight: 600; }

/* Round separator */
.round-sep {
    text-align: center;
    color: #484f58;
    font-size: 0.8em;
    margin: 16px 0 8px 0;
    position: relative;
}
.round-sep::before, .round-sep::after {
    content: '';
    position: absolute;
    top: 50%;
    width: 35%;
    height: 1px;
    background: #21262d;
}
.round-sep::before { left: 0; }
.round-sep::after { right: 0; }

/* Resolution banner */
.resolution-banner {
    background: linear-gradient(135deg, #0a3d0a, #1a4a1a);
    border: 1px solid #238636;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-top: 16px;
}
.resolution-banner h2 { color: #3fb950; margin: 0; }
.resolution-banner .score { font-size: 2em; color: #7ee787; font-weight: 700; }

/* Service cards */
.svc-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.svc-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 8px 12px;
    min-width: 120px;
    text-align: center;
}
.svc-card.running { border-color: #238636; }
.svc-card.crashed { border-color: #da3633; }
.svc-card.degraded { border-color: #d29922; }

.svc-badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.7em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.svc-badge.running { background: #0d2818; color: #3fb950; }
.svc-badge.crashed { background: #3d0a0a; color: #f85149; }
.svc-badge.degraded { background: #3d2a0a; color: #d29922; }

/* Milestone list */
.milestone-item {
    padding: 4px 8px;
    margin: 2px 0;
    border-radius: 4px;
    background: #0d2818;
    color: #3fb950;
    font-size: 0.85em;
    word-wrap: break-word;
}

/* Force HTML / Markdown containers to fit their column so long content
   (e.g. belief tracker snapshots, reward inspector tables) doesn't
   spill over into the next column. */
.gradio-container .html-container,
.gradio-container .markdown-container {
    max-width: 100%;
    overflow-x: auto;
    word-wrap: break-word;
}

/* Episode header */
.episode-header {
    background: linear-gradient(135deg, #1a0000, #0a0a2e);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
    text-align: center;
}
.episode-header h3 { color: #ff4444; margin: 0 0 4px 0; }
.episode-header .meta { color: #8b949e; font-size: 0.9em; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
}
"""


# ---- Heuristic actions for each task ----
HEURISTIC_STEPS = {
    "task1": [
        {"triage": ("get_dashboard", "diagnosis", "URGENT: nginx is DOWN. Check /var/log/nginx/error.log"),
         "diagnosis": ("", None, None), "remediation": ("", None, None)},
        {"triage": ("", None, None),
         "diagnosis": ("cat /var/log/nginx/error.log", "remediation", "nginx crashed with signal 11. Needs restart."),
         "remediation": ("", None, None)},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("systemctl restart nginx", "all", "nginx restarted. Verifying...")},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("curl http://localhost:80/health", None, None)},
    ],
    "task2": [
        {"triage": ("get_dashboard", "diagnosis", "High memory alert! Possible OOM. Check /var/log/syslog"),
         "diagnosis": ("", None, None), "remediation": ("", None, None)},
        {"triage": ("", None, None),
         "diagnosis": ("ps aux", None, None), "remediation": ("", None, None)},
        {"triage": ("", None, None),
         "diagnosis": ("cat /var/log/syslog", "remediation", "data_processor_worker PID 1000 leaking memory (2500+MB). Kill it and restart data_processor."),
         "remediation": ("", None, None)},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("kill -9 1000", None, None)},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("systemctl restart data_processor", "all", "data_processor restarted.")},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("curl http://localhost:8081/health", None, None)},
    ],
    "task3": [
        {"triage": ("get_dashboard", "diagnosis", "🚨 Redis memory at 72%! Also DB connector issues. Check Redis first."),
         "diagnosis": ("", None, None), "remediation": ("", None, None)},
        {"triage": ("", None, None),
         "diagnosis": ("cat /var/log/redis/redis.log", "triage", "Redis is fine — metrics might be stale. Checking DB connector instead."),
         "remediation": ("", None, None)},
        {"triage": ("", None, None),
         "diagnosis": ("cat /var/log/db_connector/connector.log", "remediation", "FOUND IT: authentication failed. Wrong password in /etc/app/database.yml. Replace wrong_password_123 with correct_db_pass_456. Restart: db_connector → app_server → load_balancer."),
         "remediation": ("", None, None)},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ('edit /etc/app/database.yml "wrong_password_123" "correct_db_pass_456"', "all", "Password fixed. Starting restarts...")},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("systemctl restart db_connector", None, None)},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("systemctl restart app_server", None, None)},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("systemctl restart load_balancer", None, None)},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("curl http://localhost:80/health", None, None)},
    ],
    "task4": [
        {"triage": ("get_dashboard", "diagnosis", "TWO incidents: nginx DOWN + high memory on data_processor!"),
         "diagnosis": ("", None, None), "remediation": ("", None, None)},
        {"triage": ("", None, None),
         "diagnosis": ("cat /var/log/nginx/error.log", "remediation", "nginx crashed. Restart it. Also need to investigate memory leak."),
         "remediation": ("", None, None)},
        {"triage": ("", None, None), "diagnosis": ("ps aux", None, None),
         "remediation": ("systemctl restart nginx", None, None)},
        {"triage": ("", None, None),
         "diagnosis": ("cat /var/log/syslog", "remediation", "data_processor_worker PID 1000 leaking memory. Kill it."),
         "remediation": ("", None, None)},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("kill -9 1000", None, None)},
        {"triage": ("", None, None), "diagnosis": ("", None, None),
         "remediation": ("systemctl restart data_processor", None, None)},
    ],
}


def _build_action(step_data: dict, rnd: int) -> MultiAgentAction:
    """Build MultiAgentAction from heuristic step data."""
    actions = {}
    for role in ["triage", "diagnosis", "remediation"]:
        cmd, msg_to, msg_content = step_data[role]
        message = None
        if msg_to and msg_content:
            message = Message(
                from_agent=role, to_agent=msg_to, content=msg_content,
                timestamp=datetime.now(), round_number=rnd,
            )
        actions[role] = AgentAction(command=cmd, message=message)
    return MultiAgentAction(**actions)


def _reactive_action(rnd: int) -> MultiAgentAction:
    """Fallback action that reacts to current system state.
    
    Checks for crashed services and tries to fix them — used when
    heuristic steps are exhausted (e.g., after chaos injection).
    """
    if env._system is None:
        return MultiAgentAction()
    
    # Find crashed services
    crashed = [
        name for name, svc in env._system.service_registry.services.items()
        if svc.status in ("crashed", "stopped")
    ]
    
    if not crashed:
        # Everything is running — mark as done by returning a final verification
        # Only verify once, then stop
        if reward_history and reward_history[-1] == reward_history[-2] if len(reward_history) >= 2 else False:
            # Already verified last round — do nothing to let round limit end it
            return MultiAgentAction()
        return MultiAgentAction(
            triage=AgentAction(command="get_health_summary"),
            diagnosis=AgentAction(command=""),
            remediation=AgentAction(
                command="curl http://localhost:80/health",
                message=Message(
                    from_agent="remediation", to_agent="all",
                    content="✅ All services healthy. Incident fully resolved.",
                    timestamp=datetime.now(), round_number=rnd,
                ),
            ),
        )
    
    target = crashed[0]
    return MultiAgentAction(
        triage=AgentAction(
            command="get_dashboard",
            message=Message(
                from_agent="triage", to_agent="diagnosis",
                content=f"🚨 NEW ALERT: {target} is down! Investigate immediately!",
                timestamp=datetime.now(), round_number=rnd,
            ),
        ),
        diagnosis=AgentAction(
            command="",
            message=Message(
                from_agent="diagnosis", to_agent="remediation",
                content=f"Service {target} needs restart. Please fix it.",
                timestamp=datetime.now(), round_number=rnd,
            ),
        ),
        remediation=AgentAction(
            command=f"systemctl restart {target}",
            message=Message(
                from_agent="remediation", to_agent="all",
                content=f"Restarting {target}...",
                timestamp=datetime.now(), round_number=rnd,
            ),
        ),
    )


def _format_chat_entry(role: str, command: str, message_to: str, message_content: str) -> str:
    """Format a chat entry as a polished Slack-like message bubble."""
    icon = ROLE_ICONS.get(role, "❓")
    parts = []
    if command:
        parts.append(f'<div class="cmd-block">{command}</div>')
    if message_to and message_content:
        parts.append(
            f'<div class="msg-bubble">'
            f'<span class="msg-target">→ @{message_to}</span> {message_content}'
            f'</div>'
        )
    content = "\n".join(parts) if parts else '<div style="color:#484f58;font-style:italic">(no action)</div>'
    return (
        f'<div class="agent-msg {role}">'
        f'<span class="agent-name {role}">{icon} @{role.capitalize()}</span>'
        f'{content}'
        f'</div>'
    )


def _service_status_html(system_snapshot: dict) -> str:
    """Build HTML card grid for service status panel."""
    if not system_snapshot:
        return '<div style="color:#484f58;text-align:center;padding:16px"><em>No data</em></div>'
    services = system_snapshot.get("service_registry", {}).get("services", {})
    cards = []
    for name, svc in sorted(services.items()):
        status = svc.get("status", "unknown")
        icon = "🟢" if status == "running" else "🔴" if status == "crashed" else "🟡"
        badge_cls = status if status in ("running", "crashed", "degraded") else "degraded"
        port = svc.get("port", "?")
        deps = svc.get("depends_on", [])
        dep_html = ""
        if deps:
            dep_arrows = " → ".join(deps)
            dep_html = f'<div style="font-size:0.7em;color:#484f58;margin-top:2px">↳ {dep_arrows}</div>'
        cards.append(
            f'<div class="svc-card {badge_cls}">'
            f'<div style="font-size:1.4em">{icon}</div>'
            f'<div><strong style="color:#c9d1d9">{name}</strong></div>'
            f'<div><span class="svc-badge {badge_cls}">{status}</span></div>'
            f'<div style="font-size:0.8em;color:#8b949e">:{port}</div>'
            f'{dep_html}'
            f'</div>'
        )
    return f'<div class="svc-grid">{"".join(cards)}</div>'


def _milestone_html(milestones: list) -> str:
    """Build styled milestone checklist."""
    if not milestones:
        return '<div style="color:#484f58;font-style:italic;padding:8px">No milestones yet</div>'
    items = [f'<div class="milestone-item">✅ {m}</div>' for m in milestones]
    return "".join(items)


def _empty_card(title: str, hint: str) -> str:
    return (
        "<div style='background:#161b22;border:1px dashed #30363d;border-radius:8px;padding:10px'>"
        f"<div style='color:#c9d1d9;font-weight:700;margin-bottom:4px'>{title}</div>"
        f"<div style='color:#8b949e;font-size:0.85em;font-style:italic'>{hint}</div>"
        "</div>"
    )


COMPONENT_LABELS = {
    "milestone_credit": "Milestone credit",
    "penalty_total": "Penalties",
    "communication_bonus": "Comm bonus",
    "raw_score": "Raw score",
    "final_score": "Final (clamped)",
}


def _reward_inspector_html(obs) -> str:
    """Render current reward component breakdown."""
    if obs is None:
        return _empty_card(
            "Reward Inspector",
            "Start an episode and step a round — components will appear here.",
        )
    metadata = getattr(obs, "metadata", {}) or {}
    components = metadata.get("reward_components", {})
    penalties = metadata.get("penalty_reasons", [])
    if not components:
        return _empty_card(
            "Reward Inspector",
            "Step at least one round to populate reward components.",
        )

    def _row(k: str, v: float) -> str:
        label = COMPONENT_LABELS.get(k, k)
        sign_color = "#3fb950" if v >= 0 and k != "penalty_total" else "#f85149" if k == "penalty_total" and v > 0 else "#c9d1d9"
        return (
            f"<tr><td style='padding:3px 8px'>{label}</td>"
            f"<td style='padding:3px 8px;text-align:right;color:{sign_color}'><code>{v:.3f}</code></td></tr>"
        )

    rows = [_row(k, v) for k, v in components.items()]
    penalty_text = ", ".join(penalties) if penalties else "none"
    return (
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px'>"
        "<div style='color:#c9d1d9;font-weight:700;margin-bottom:6px'>Reward Inspector</div>"
        "<table style='width:100%;color:#8b949e;font-size:0.85em'>"
        + "".join(rows)
        + "</table>"
        f"<div style='margin-top:6px;color:#8b949e;font-size:0.8em'><b>penalties:</b> {penalty_text}</div>"
        "</div>"
    )


def _deception_banner_html() -> str:
    """Show a banner when phantom alerts are active or detected (Theory of Mind cue)."""
    global env
    tracker = getattr(env, "_belief_tracker", None) if env else None
    if not tracker:
        return ""
    try:
        dec = tracker.get_deception_score()
    except Exception:
        return ""
    total = dec.get("phantom_alerts_total", 0)
    if not total:
        return ""
    detected = dec.get("phantom_alerts_detected", 0)
    chased = dec.get("phantom_alerts_chased", 0)
    score = dec.get("deception_score", 0.0)

    if detected > 0:
        bg = "#0a2a14"
        border = "#3fb950"
        title = "✅ Phantom alert detected — agents pushed back"
    elif chased > 0:
        bg = "#3d1f0a"
        border = "#d29922"
        title = "⚠️ Agents are chasing a phantom alert"
    else:
        bg = "#1a1a2e"
        border = "#58a6ff"
        title = "🧠 Phantom alert active — Theory of Mind under test"

    return (
        f"<div style='background:{bg};border-left:4px solid {border};border-radius:6px;padding:8px 12px;margin:6px 0'>"
        f"<div style='color:#c9d1d9;font-weight:700;font-size:0.9em'>{title}</div>"
        f"<div style='color:#8b949e;font-size:0.78em;margin-top:2px'>"
        f"detected {detected}/{total} · chased {chased} · deception score <code>{score:.2f}</code>"
        "</div></div>"
    )


def _round_trace_html() -> str:
    """Render round-by-round interaction trace."""
    if not round_trace:
        return _empty_card(
            "Live Incident Playback",
            "Each row will show what Triage / Diagnosis / Remediation did and the resulting reward.",
        )
    rows = []
    for item in round_trace[-12:]:
        rows.append(
            "<tr>"
            f"<td>{item['round']}</td>"
            f"<td><code>{item['triage'] or '-'}</code></td>"
            f"<td><code>{item['diagnosis'] or '-'}</code></td>"
            f"<td><code>{item['remediation'] or '-'}</code></td>"
            f"<td>{item['reward']:.3f}</td>"
            "</tr>"
        )
    return (
        "<div style='background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px'>"
        "<div style='color:#c9d1d9;font-weight:700;margin-bottom:6px'>Live Incident Playback</div>"
        "<table style='width:100%;font-size:0.8em;color:#8b949e'>"
        "<tr><th>R</th><th>Triage</th><th>Diagnosis</th><th>Remediation</th><th>Reward</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )


def _reward_plot(rewards: list) -> plt.Figure:
    """Create a dark-themed reward progress plot with gradient fill and milestone markers."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(6, 2.5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    x = list(range(1, len(rewards) + 1))
    y = rewards

    # Main line
    ax.plot(x, y, color='#58a6ff', linewidth=2.5, marker='o', markersize=5,
            markerfacecolor='#58a6ff', markeredgecolor='#0d1117', markeredgewidth=1.5, zorder=3)

    # Gradient fill under the curve
    if len(x) > 1:
        ax.fill_between(x, y, alpha=0.15, color='#58a6ff')
        # Subtle gradient effect with multiple fills
        for alpha_val, offset in [(0.08, 0.02), (0.04, 0.05)]:
            shifted = [max(0, v - offset) for v in y]
            ax.fill_between(x, shifted, alpha=alpha_val, color='#58a6ff')

    # Milestone markers at reward jumps
    for i in range(1, len(y)):
        if y[i] - y[i - 1] > 0.05:
            ax.annotate('★', xy=(x[i], y[i]), fontsize=14, color='#FFD700',
                        ha='center', va='bottom', fontweight='bold')

    ax.set_xlabel("Round", color='#8b949e', fontsize=10)
    ax.set_ylabel("Team Reward", color='#8b949e', fontsize=10)
    ax.set_title("Reward Progress", color='#c9d1d9', fontsize=12, fontweight='bold', pad=10)
    ax.set_ylim(-0.02, 1.05)
    ax.tick_params(colors='#484f58')
    ax.spines['bottom'].set_color('#21262d')
    ax.spines['left'].set_color('#21262d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.1, color='#30363d')

    plt.tight_layout()
    return fig


def _comm_flow_graph(messages: list) -> plt.Figure:
    """Create a communication flow graph showing agent interactions.

    Three nodes (Triage, Diagnosis, Remediation) in a triangle.
    Directed arrows show message flow with thickness = message count.
    Arrow color matches the source agent.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Node positions (triangle)
    positions = {
        "triage": (0.5, 0.9),
        "diagnosis": (0.15, 0.2),
        "remediation": (0.85, 0.2),
    }

    node_colors = {
        "triage": "#FFD700",
        "diagnosis": "#00CED1",
        "remediation": "#32CD32",
    }

    node_icons = {
        "triage": "🚨",
        "diagnosis": "🔎",
        "remediation": "🛠️",
    }

    # Count messages between agents
    flow = {}  # {(from, to): count}
    for msg in messages:
        key = (msg.from_agent, msg.to_agent)
        flow[key] = flow.get(key, 0) + 1
        # Also count "all" messages
        if msg.to_agent == "all":
            for target in ["triage", "diagnosis", "remediation"]:
                if target != msg.from_agent:
                    k2 = (msg.from_agent, target)
                    flow[k2] = flow.get(k2, 0) + 1

    # Draw edges (curved arrows)
    for (src, dst), count in flow.items():
        if src not in positions or dst not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[dst]

        # Arrow thickness based on count
        lw = min(1 + count * 1.5, 6)
        alpha = min(0.4 + count * 0.15, 1.0)

        # Arrow color based on source
        color = node_colors.get(src, "#58a6ff")

        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=lw,
                alpha=alpha,
                connectionstyle="arc3,rad=0.2",
                mutation_scale=15,
            ),
        )

        # Message count label on the edge
        mx = (x1 + x2) / 2 + 0.05 * (y2 - y1)
        my = (y1 + y2) / 2 - 0.05 * (x2 - x1)
        ax.text(mx, my, str(count), fontsize=10, color=color, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#0d1117', edgecolor=color, alpha=0.8))

    # Draw nodes (circles with labels)
    for role, (x, y) in positions.items():
        color = node_colors[role]
        icon = node_icons[role]

        circle = plt.Circle((x, y), 0.08, color=color, alpha=0.15, zorder=2)
        ax.add_patch(circle)
        circle_border = plt.Circle((x, y), 0.08, fill=False, color=color, linewidth=2, zorder=3)
        ax.add_patch(circle_border)

        ax.text(x, y + 0.01, icon, fontsize=20, ha='center', va='center', zorder=4)

        ax.text(x, y - 0.12, role.capitalize(), fontsize=11, color=color,
                ha='center', va='center', fontweight='bold', zorder=4)

    total_msgs = sum(flow.values())
    ax.set_title(f"Communication Flow ({total_msgs} messages)",
                 color='#c9d1d9', fontsize=13, fontweight='bold', pad=15)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    return fig


def _comm_timeline(messages: list, max_rounds: int) -> plt.Figure:
    """Create a timeline showing when agents communicated and to whom."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    roles = ["triage", "diagnosis", "remediation"]
    role_y = {"triage": 2, "diagnosis": 1, "remediation": 0}
    role_colors = {"triage": "#FFD700", "diagnosis": "#00CED1", "remediation": "#32CD32"}

    # Draw horizontal lanes
    for role, y in role_y.items():
        ax.axhline(y=y, color='#21262d', linewidth=0.5, linestyle='--')
        ax.text(-0.5, y, f"{ROLE_ICONS.get(role, '')} {role.capitalize()}",
                fontsize=10, color=role_colors[role], va='center', fontweight='bold')

    # Draw messages as arrows
    for msg in messages:
        src = msg.from_agent
        dst = msg.to_agent
        rnd = msg.round_number

        if src not in role_y:
            continue

        y_src = role_y[src]
        color = role_colors.get(src, '#58a6ff')

        if dst == "all":
            for target in roles:
                if target != src and target in role_y:
                    y_dst = role_y[target]
                    ax.annotate("", xy=(rnd, y_dst), xytext=(rnd, y_src),
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, alpha=0.6))
        elif dst in role_y:
            y_dst = role_y[dst]
            ax.annotate("", xy=(rnd, y_dst), xytext=(rnd, y_src),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2, alpha=0.8))

        # Message dot at source
        ax.scatter(rnd, y_src, s=60, color=color, zorder=5, edgecolors='#0d1117', linewidths=1)

    ax.set_xlabel("Round", color='#8b949e', fontsize=10)
    ax.set_title("Communication Timeline", color='#c9d1d9', fontsize=12, fontweight='bold')
    ax.set_xlim(-1, max(max_rounds, 5) + 0.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#21262d')
    ax.tick_params(colors='#484f58')

    plt.tight_layout()
    return fig


# ---- Tab 1: War Room ----

def _belief_state_html():
    global env
    if env and hasattr(env, '_belief_tracker') and env._belief_tracker:
        snapshot = env._belief_tracker.get_snapshot()
        if snapshot.get("round", 0) == 0 and not snapshot.get("agents", {}):
            return _empty_card(
                "🧠 Theory of Mind Tracker",
                "Every round, this engine records what each agent believes about every "
                "service and compares it against ground truth. Agents that push back on "
                "false beliefs score on the Deception Resistance metric. Start an "
                "episode to see beliefs evolve.",
            )
        banner = _deception_banner_html()
        return banner + env._belief_tracker.format_html()
    return _empty_card(
        "🧠 Theory of Mind Tracker",
        "No Belief State Available. Start an episode.",
    )


def _pushback_summary_html() -> str:
    """Sticky top-of-chat banner summarizing Theory of Mind pushback events.

    Surfaces the single signal judges care most about: did the team detect
    a false belief and push back? This banner counts pushback events and
    flips from dormant (grey) to celebratory (green) the moment one fires.
    """
    global env
    tracker = getattr(env, "_belief_tracker", None) if env else None
    if not tracker:
        return ""
    try:
        snap = tracker.get_snapshot()
    except Exception:
        return ""
    tom_events = snap.get("tom_events", []) or []
    count = len(tom_events)
    if count == 0:
        return (
            '<div style="background:#161b22;border-left:4px solid #30363d;'
            'border-radius:6px;padding:8px 14px;margin:0 0 10px 0">'
            '<span style="color:#8b949e;font-weight:700">🧠 Theory of Mind watch</span>'
            '<span style="color:#484f58;margin-left:10px;font-size:0.85em">'
            '0 pushbacks so far — will light up when an agent contradicts a false belief'
            '</span></div>'
        )
    # One or more pushbacks — celebrate and list them briefly.
    bullets = []
    for ev in tom_events[:4]:
        detector = (ev.get("detector") or "?").capitalize()
        target = (ev.get("target") or "?").capitalize()
        entity = ev.get("entity") or "?"
        bullets.append(
            f'<li style="margin:2px 0"><b style="color:#bc8cff">{detector}</b> '
            f'→ <b>{target}</b>: pushback on <code>{entity}</code></li>'
        )
    more = ""
    if count > 4:
        more = f'<li style="color:#8b949e;font-style:italic">…and {count - 4} more</li>'
    return (
        '<div style="background:linear-gradient(135deg,#0a2a14,#0a1f1a);'
        'border-left:4px solid #3fb950;border-radius:6px;padding:10px 14px;margin:0 0 10px 0">'
        f'<div style="color:#3fb950;font-weight:700;font-size:1.0em">'
        f'🧠 Theory of Mind moments: {count}'
        '</div>'
        '<ul style="color:#c9d1d9;font-size:0.85em;margin:4px 0 0 0;padding-left:20px">'
        + "".join(bullets) + more +
        '</ul></div>'
    )


def start_episode(task_id: str, seed: int, use_agent_mode: bool = False,
                  model_name: str = "", api_base_url: str = ""):
    global env, current_obs, round_num, chat_history, reward_history, milestone_list, thought_history, round_trace
    global live_runner, agent_mode_enabled

    task_key = _parse_task_key(task_id)
    env = WarRoomEnvironment()
    current_obs = env.reset(task_id=task_key, seed=seed)
    round_num = 0
    chat_history = []
    reward_history = []
    milestone_list = []
    thought_history = []
    round_trace = []

    agent_mode_enabled = bool(use_agent_mode)
    if agent_mode_enabled:
        cfg = LiveAgentConfig()
        if model_name and model_name.strip():
            cfg.model_name = model_name.strip()
        if api_base_url and api_base_url.strip():
            cfg.api_base_url = api_base_url.strip()
            cfg.__post_init__()  # re-normalize if user changed the URL
        if not cfg.is_ready():
            agent_mode_enabled = False
            chat_history.append(
                '<div class="agent-msg" style="border-left-color:#f85149;background:#3d0a0a">'
                '<span style="color:#f85149;font-weight:700">⚠️ Agent Mode unavailable</span>'
                '<div class="msg-bubble" style="border-left-color:#f85149">'
                'HF_TOKEN / API_KEY not set. Falling back to heuristic mode. '
                'Set a token in the environment or Space secrets to use the LLM.'
                '</div></div>'
            )
        else:
            live_runner = LiveAgentRunner(cfg)
            live_runner.reset()
            is_trained = ".endpoints.huggingface.cloud" in cfg.api_base_url
            banner_color = "#3fb950" if is_trained else "#58a6ff"
            banner_bg = "#0a2a1a" if is_trained else "#0a1a2a"
            label = "🎯 Trained adapter" if is_trained else "🤖 Base model"
            chat_history.append(
                f'<div class="agent-msg" style="border-left-color:{banner_color};background:{banner_bg}">'
                f'<span style="color:{banner_color};font-weight:700">{label} active</span>'
                f'<div class="msg-bubble" style="border-left-color:{banner_color}">'
                f'Model <code>{cfg.model_name}</code> via <code>{cfg.api_base_url}</code>. '
                f'Each round queries the LLM for all three agents.'
                f'</div></div>'
            )

    task_name = current_obs.metadata.get("task_name", task_key)
    difficulty = current_obs.metadata.get("difficulty", "?")
    max_rounds = current_obs.metadata.get("max_rounds", "?")

    header = (
        f'<div class="episode-header">'
        f'<h3>🔧 INCIDENT ACTIVE</h3>'
        f'<div class="meta">{task_name} &middot; {difficulty} &middot; Max {max_rounds} rounds</div>'
        f'</div>'
    )
    chat_history.append(header)

    system_html = _service_status_html(env.state.simulated_system)
    chat_html = _pushback_summary_html() + "\n".join(chat_history)
    fig = _reward_plot([0.0])
    empty_flow = _comm_flow_graph([])
    empty_timeline = _comm_timeline([], 10)
    thought_html = _empty_card(
        "💭 Agent Brain Scanner",
        "Enable Agent Mode and run a round to see each agent's structured "
        "rationale (thought → command → message). Base models often show "
        "panic heuristics; trained agents show evidence checks.",
    )

    return chat_html, system_html, fig, "Episode started. Click 'Next Round' to step.", _milestone_html([]), empty_flow, empty_timeline, _belief_state_html(), thought_html, _reward_inspector_html(current_obs), _round_trace_html()


def next_round(task_id: str):
    global current_obs, round_num, chat_history, reward_history, milestone_list, thought_history, round_trace

    task_key = _parse_task_key(task_id)

    if current_obs is None:
        return "\n".join(chat_history), "<em>Start an episode first</em>", _reward_plot([0]), "Start an episode first.", _milestone_html([]), _comm_flow_graph([]), _comm_timeline([], 10), _belief_state_html(), "".join(thought_history), _reward_inspector_html(current_obs), _round_trace_html()

    if current_obs.done:
        messages = env._channel.get_full_history() if env._channel else []
        max_r = env._max_rounds if hasattr(env, '_max_rounds') else 10
        return "\n".join(chat_history), _service_status_html(env.state.simulated_system), _reward_plot(reward_history), "Episode complete!", _milestone_html(milestone_list), _comm_flow_graph(messages), _comm_timeline(messages, max_r), _belief_state_html(), "".join(thought_history), _reward_inspector_html(current_obs), _round_trace_html()

    steps = HEURISTIC_STEPS.get(task_key, HEURISTIC_STEPS["task1"])

    # Choose action source: LLM (agent mode) vs scripted heuristic
    if agent_mode_enabled and live_runner is not None:
        try:
            action = live_runner.step(
                round_num=round_num + 1,
                triage_obs=current_obs.triage.text,
                diagnosis_obs=current_obs.diagnosis.text,
                remediation_obs=current_obs.remediation.text,
            )
        except Exception as exc:
            # On any LLM failure, fall back to scripted heuristic for this round
            chat_history.append(
                f'<div class="agent-msg" style="border-left-color:#f85149;background:#3d0a0a">'
                f'<span style="color:#f85149;font-weight:700">⚠️ Agent Mode error</span>'
                f'<div class="msg-bubble" style="border-left-color:#f85149">'
                f'{exc}. Using scripted heuristic for round {round_num + 1}.'
                f'</div></div>'
            )
            if round_num >= len(steps):
                action = _reactive_action(round_num + 1)
            else:
                action = _build_action(steps[round_num], round_num + 1)
    elif round_num >= len(steps):
        # Reactive fallback: check for crashed services and try to fix them
        action = _reactive_action(round_num + 1)
    else:
        step_data = steps[round_num]
        action = _build_action(step_data, round_num + 1)

    current_obs = env.step(action)
    round_num += 1
    reward_history.append(current_obs.team_reward)
    round_trace.append({
        "round": round_num,
        "triage": action.triage.command,
        "diagnosis": action.diagnosis.command,
        "remediation": action.remediation.command,
        "reward": current_obs.team_reward,
    })

    # Round separator with timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    chat_history.append(
        f'<div class="round-sep">── Round {round_num} &middot; {ts} ──</div>'
    )

    # Add agent actions to chat
    for role in ["triage", "diagnosis", "remediation"]:
        a = getattr(action, role)
        if a.command or a.message:
            msg_to = a.message.to_agent if a.message else None
            msg_content = a.message.content if a.message else None
            chat_history.append(_format_chat_entry(role, a.command, msg_to, msg_content))
        if hasattr(a, 'thought') and a.thought:
            thought_history.append(
                f'<div style="font-family: monospace; font-size: 0.85em; color: #a371f7; background: #2a1b41; padding: 8px; border-radius: 6px; margin-bottom: 6px; border-left: 3px solid #8957e5;">'
                f'<strong style="color: #bc8cff;">[{role.upper()}]</strong><br/>{a.thought}'
                f'</div>'
            )

    # Show executive/chaos monkey messages from the channel
    if env._channel:
        channel_msgs = env._channel.get_full_history()
        for msg in channel_msgs:
            if msg.round_number == round_num and msg.from_agent in ("executive", "chaos_monkey"):
                color = "#f97316" if msg.from_agent == "executive" else "#f85149"
                icon = "👔" if msg.from_agent == "executive" else "🐒"
                label = "Executive" if msg.from_agent == "executive" else "ChaosMonkey"
                chat_history.append(
                    f'<div class="agent-msg" style="border-left-color:{color};background:#1a1000">'
                    f'<span style="color:{color};font-weight:700">{icon} @{label}</span>'
                    f'<div class="msg-bubble" style="border-left-color:{color}">{msg.content}</div>'
                    f'</div>'
                )

    # Theory of Mind: highlight pushback events from this round. These fire
    # when an agent explicitly contradicts another agent's false belief (e.g.
    # "Redis metrics are stale, the real issue is the DB password"). This is
    # the single biggest behaviour the environment is designed to produce.
    if env and getattr(env, "_belief_tracker", None):
        tom_events = env._belief_tracker.get_snapshot().get("tom_events", [])
        for ev in tom_events:
            if ev.get("round") == round_num:
                detector = ev.get("detector", "?").capitalize()
                target = ev.get("target", "?").capitalize()
                entity = ev.get("entity", "?")
                quote = (ev.get("message", "") or "").strip()[:180]
                chat_history.append(
                    '<div class="agent-msg" style="border-left-color:#a371f7;'
                    'background:linear-gradient(135deg,#2a1b41,#1a0a2a)">'
                    '<span style="color:#bc8cff;font-weight:700">🧠 Theory of Mind moment</span>'
                    '<div class="msg-bubble" style="border-left-color:#a371f7">'
                    f'<b>{detector}</b> pushed back on <b>{target}</b>\'s belief about '
                    f'<code>{entity}</code>. '
                    f'<i>&ldquo;{quote}&rdquo;</i>'
                    '</div></div>'
                )

    # Update milestones
    if current_obs.done and "milestones_achieved" in current_obs.metadata:
        milestone_list = current_obs.metadata["milestones_achieved"]
    elif hasattr(env, '_grader') and env._grader:
        milestone_list = sorted(env._grader.achieved)

    # Status message
    status = f"Round {round_num} | Reward: {current_obs.team_reward:.3f}"
    if current_obs.done:
        score = current_obs.metadata.get("score", current_obs.team_reward)
        milestone_count = len(milestone_list)
        chat_history.append(
            f'<div class="resolution-banner">'
            f'<h2>🎉 INCIDENT RESOLVED</h2>'
            f'<div class="score">{score:.3f}</div>'
            f'<div style="color:#8b949e">{round_num} rounds &middot; {milestone_count} milestones</div>'
            f'</div>'
        )
        status = f"✅ RESOLVED — Score: {score:.3f}"

    chat_html = _pushback_summary_html() + "\n".join(chat_history)
    system_html = _service_status_html(env.state.simulated_system)
    fig = _reward_plot(reward_history)
    messages = env._channel.get_full_history() if env._channel else []
    max_r = env._max_rounds if hasattr(env, '_max_rounds') else 10
    flow_fig = _comm_flow_graph(messages)
    timeline_fig = _comm_timeline(messages, max_r)

    return chat_html, system_html, fig, status, _milestone_html(milestone_list), flow_fig, timeline_fig, _belief_state_html(), "".join(thought_history), _reward_inspector_html(current_obs), _round_trace_html()


def auto_play(task_id: str, seed: int, use_agent_mode: bool = False,
              model_name: str = "", api_base_url: str = ""):
    """Run entire episode automatically."""
    result = start_episode(task_id, seed, use_agent_mode, model_name, api_base_url)

    for _ in range(30):  # Max iterations
        if current_obs and current_obs.done:
            break
        result = next_round(task_id)

    return result


def inject_chaos():
    """Inject a random failure mid-episode."""
    global current_obs
    if env._system is None:
        return "\n".join(chat_history), _service_status_html({}), _reward_plot(reward_history), "Start an episode first!", _milestone_html([]), _comm_flow_graph([]), _comm_timeline([], 10), _belief_state_html(), "".join(thought_history), _reward_inspector_html(current_obs), _round_trace_html()

    result = env.inject_chaos()

    # Add chaos event to chat
    chat_history.append(
        f'<div class="agent-msg" style="border-left-color:#f85149;background:#3d0a0a">'
        f'<span style="color:#f85149;font-weight:700">🐒 @ChaosMonkey</span>'
        f'<div style="color:#f85149">{result}</div>'
        f'</div>'
    )

    system_html = _service_status_html(env.state.simulated_system)
    messages = env._channel.get_full_history() if env._channel else []
    max_r = env._max_rounds if hasattr(env, '_max_rounds') else 10

    return "\n".join(chat_history), system_html, _reward_plot(reward_history), f"💥 {result}", _milestone_html(milestone_list), _comm_flow_graph(messages), _comm_timeline(messages, max_r), _belief_state_html(), "".join(thought_history), _reward_inspector_html(current_obs), _round_trace_html()


def send_judge_message(msg: str, target_agent: str):
    """Inject a custom message from the Judge (CEO) mid-episode."""
    if env._system is None or not msg.strip():
        messages = env._channel.get_full_history() if getattr(env, '_channel', None) else []
        max_r = env._max_rounds if hasattr(env, '_max_rounds') else 10
        return "\n".join(chat_history), _service_status_html({}), _reward_plot(reward_history), "Start an episode first!", _milestone_html([]), _comm_flow_graph(messages), _comm_timeline(messages, max_r), _belief_state_html(), "".join(thought_history), _reward_inspector_html(current_obs), _round_trace_html(), ""

    env.inject_external_message(msg.strip(), from_agent="executive", to_agent=target_agent)

    chat_history.append(
        f'<div class="agent-msg" style="border-left-color:#f97316;background:#1a1000">'
        f'<span style="color:#f97316;font-weight:700">👔 @CEO (Judge)</span>'
        f'<div class="msg-bubble" style="border-left-color:#f97316">{msg.strip()}</div>'
        f'</div>'
    )

    system_html = _service_status_html(env.state.simulated_system)
    messages = env._channel.get_full_history() if env._channel else []
    max_r = env._max_rounds if hasattr(env, '_max_rounds') else 10

    return "\n".join(chat_history), system_html, _reward_plot(reward_history), f"Sent message to {target_agent}.", _milestone_html(milestone_list), _comm_flow_graph(messages), _comm_timeline(messages, max_r), _belief_state_html(), "".join(thought_history), _reward_inspector_html(current_obs), _round_trace_html(), ""


# ---- Tab 2: Training Curves ----

def _dark_training_plot(ax, fig):
    """Apply dark theme styling to a training plot axis."""
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.tick_params(colors='#484f58')
    ax.spines['bottom'].set_color('#21262d')
    ax.spines['left'].set_color('#21262d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.1, color='#30363d')
    ax.xaxis.label.set_color('#8b949e')
    ax.yaxis.label.set_color('#8b949e')
    ax.title.set_color('#c9d1d9')


def load_training_metrics():
    """Load and plot training metrics with dark-themed charts."""
    plt.style.use('dark_background')
    metrics_path = "outputs/war_room_training/metrics.json"
    if not os.path.exists(metrics_path):
        return None, None, None, None, "No training data found. Run training first."

    with open(metrics_path) as f:
        metrics = json.load(f)

    episodes = metrics["episode"]
    rewards = metrics["team_reward"]
    rounds_used = metrics["rounds_used"]
    milestones = metrics["milestones_achieved"]
    tasks = metrics["task"]

    # Plot 1: Reward curve
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    _dark_training_plot(ax1, fig1)
    ax1.plot(episodes, rewards, color='#58a6ff', marker='o', markersize=3, linewidth=1, alpha=0.7)
    ax1.fill_between(episodes, rewards, alpha=0.1, color='#58a6ff')
    if len(rewards) >= 5:
        rolling = [sum(rewards[max(0, i - 4):i + 1]) / min(i + 1, 5) for i in range(len(rewards))]
        ax1.plot(episodes, rolling, color='#f85149', linewidth=2, label='5-ep rolling avg')
        ax1.legend(facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    # Milestone markers on reward jumps
    for i in range(1, len(rewards)):
        if rewards[i] - rewards[i - 1] > 0.1:
            ax1.annotate('★', xy=(episodes[i], rewards[i]), fontsize=12, color='#FFD700',
                         ha='center', va='bottom')
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Team Reward")
    ax1.set_title("Reward Over Training Episodes", fontweight='bold')
    ax1.set_ylim(0, 1)
    plt.tight_layout()

    # Plot 2: Rounds to resolve
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    _dark_training_plot(ax2, fig2)
    ax2.plot(episodes, rounds_used, color='#3fb950', marker='o', markersize=3, linewidth=1)
    ax2.fill_between(episodes, rounds_used, alpha=0.1, color='#3fb950')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rounds Used")
    ax2.set_title("Rounds to Resolve (↓ = more efficient)", fontweight='bold')
    plt.tight_layout()

    # Plot 3: Milestones
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    _dark_training_plot(ax3, fig3)
    ax3.bar(episodes, milestones, color='#bc8cff', alpha=0.7, edgecolor='#8957e5', linewidth=0.5)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Milestones")
    ax3.set_title("Milestones Achieved Per Episode", fontweight='bold')
    plt.tight_layout()

    # Plot 4: By task
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    _dark_training_plot(ax4, fig4)
    task_rewards = {}
    for t, r in zip(tasks, rewards):
        task_rewards.setdefault(t, []).append(r)
    colors = ['#58a6ff', '#3fb950', '#d29922', '#f85149']
    for idx, (tid, tr) in enumerate(sorted(task_rewards.items())):
        c = colors[idx % len(colors)]
        ax4.bar(tid, sum(tr) / len(tr), alpha=0.8, color=c, edgecolor=c, linewidth=0.5)
    ax4.set_xlabel("Task")
    ax4.set_ylabel("Avg Reward")
    ax4.set_title("Average Reward by Task", fontweight='bold')
    ax4.set_ylim(0, 1)
    plt.tight_layout()

    avg = sum(rewards) / len(rewards) if rewards else 0
    summary = f"Episodes: {len(episodes)} | Avg Reward: {avg:.3f} | Best: {max(rewards):.3f}"

    return fig1, fig2, fig3, fig4, summary


def run_training_sim(episodes: int):
    """Run training simulation and refresh charts."""
    import subprocess
    subprocess.run([sys.executable, "round2/war_room/train.py", "--episodes", str(int(episodes))],
                   env={**os.environ, "PYTHONPATH": "."}, capture_output=True)
    return load_training_metrics()


# ---- Build the app ----

def build_app():
    # Pass css/theme via constructor for Gradio <6.0 compat; they also work in launch() for 6.0+
    try:
        app_kwargs = dict(title="Multi-Agent Incident War Room", css=CUSTOM_CSS,
                          theme=gr.themes.Base(primary_hue="red", neutral_hue="slate"))
    except Exception:
        app_kwargs = dict(title="Multi-Agent Incident War Room")
    with gr.Blocks(**app_kwargs) as app:

        # Header banner
        gr.HTML('''
        <div class="war-room-header">
            <h1>🔧 Multi-Agent Incident War Room</h1>
            <p>Three AI agents with partial observability cooperate through a shared communication channel to diagnose and fix production incidents</p>
        </div>
        ''')

        # How it works accordion
        with gr.Accordion("💡 How it works", open=False):
            gr.Markdown("""
Each episode simulates a production incident. Three specialized agents — **Triage**, **Diagnosis**, and **Remediation** — each see different parts of the system and must communicate to resolve the issue.

1. **Select a task** from the dropdown (each has a different difficulty and scenario)
2. **Start the episode** to initialize the simulated infrastructure
3. **Step through rounds** to watch the agents coordinate, or use **Auto-Play** to run the full episode
4. Watch the **reward curve** climb as milestones are achieved
            """)

        with gr.Tabs():
            # ---- Tab 1: War Room ----
            with gr.Tab("🔧 War Room"):
                # Controls row — compact
                with gr.Row():
                    task_dropdown = gr.Dropdown(
                        choices=list(TASK_DESCRIPTIONS.values()),
                        value=TASK_DESCRIPTIONS["task1"],
                        label="Incident Scenario",
                        scale=3,
                    )
                    seed_input = gr.Number(value=42, label="Seed", precision=0, scale=1)
                    start_btn = gr.Button("▶️ Start", variant="primary", scale=1)
                    next_btn = gr.Button("⏭️ Next", scale=1)
                    auto_btn = gr.Button("⏩ Auto", variant="secondary", scale=1)
                    chaos_btn = gr.Button("💥 INJECT CHAOS", variant="stop", scale=1)
                
                # Live Judge Mode Row
                with gr.Row():
                    judge_input = gr.Textbox(
                        label="👔 Live Judge Mode (Act as CEO)",
                        placeholder="Type a message to interrupt the agents (e.g. 'Why is the site down? Fix it NOW!')",
                        scale=5,
                    )
                    judge_target = gr.Dropdown(
                        choices=["all", "triage", "diagnosis", "remediation"],
                        value="all",
                        label="Target Agent",
                        scale=1,
                    )
                    judge_btn = gr.Button("📨 Send as CEO", variant="primary", scale=1)

                # Agent Mode controls: toggle scripted vs LLM-driven rollout
                with gr.Accordion("🤖 Agent Mode (live LLM rollout)", open=True):
                    gr.Markdown(
                        "**Default (unchecked):** Scripted heuristic — acts like a "
                        "perfectly-trained agent. Resolves in 4-6 rounds with correct format.\n\n"
                        "**Agent Mode (checked):** Live LLM rollout using the base "
                        "**🤖 Qwen 7B** model. Watch it kill healthy nginx, follow "
                        "executive panic, and loop — the untrained baseline the GRPO "
                        "run is measured against.\n"
                    )
                    gr.Markdown(
                        "> ℹ️ The trained adapter is published at "
                        "[brodie1of1/war-room-grpo-adapter]"
                        "(https://huggingface.co/brodie1of1/war-room-grpo-adapter). "
                        "Loading requires peft + 15GB base weights, which exceeds "
                        "free Space hardware. See README for local-loading instructions."
                    )
                    with gr.Row():
                        agent_mode_toggle = gr.Checkbox(
                            value=False,
                            label="Enable Agent Mode (live LLM)",
                            scale=1,
                        )
                        agent_preset = gr.Radio(
                            choices=["🤖 Base Qwen 7B", "🎯 Trained (local MLX)"],
                            value="🤖 Base Qwen 7B",
                            label="Preset",
                            scale=2,
                            info="Trained preset requires local MLX server — run bash scripts/run_mlx_server.sh on your M-series Mac first.",
                        )
                    with gr.Row():
                        model_name_input = gr.Textbox(
                            value="Qwen/Qwen2.5-7B-Instruct",
                            label="Model",
                            placeholder="Qwen/Qwen2.5-7B-Instruct",
                            scale=2,
                        )
                        api_base_url_input = gr.Textbox(
                            value="",
                            label="API Base URL (leave blank for default)",
                            placeholder="https://router.huggingface.co/v1",
                            scale=3,
                        )

                    def _apply_preset(preset: str):
                        """Auto-fill model/URL fields when the preset changes."""
                        if "Trained" in preset:
                            # Local MLX server (Apple Silicon). Start it with:
                            #   bash scripts/run_mlx_server.sh
                            # Then the Gradio UI connects at localhost:8080.
                            return (
                                "brodie1of1/war-room-7b-merged",
                                "http://localhost:8080/v1",
                            )
                        return ("Qwen/Qwen2.5-7B-Instruct", "")

                    agent_preset.change(
                        _apply_preset,
                        inputs=[agent_preset],
                        outputs=[model_name_input, api_base_url_input],
                    )

                status_text = gr.Textbox(label="Status", interactive=False, max_lines=1)

                # ---- Main dashboard ----
                # Clean vertical flow: chat at top, status cards in a row,
                # plots in a row, deep-dive views hidden in an accordion.
                # This prevents the old 4-column-stacked-widget overlap.

                # Row 1: Live agent chat (full width, its own scroll container)
                chat_display = gr.HTML(
                    label="Agent Chat",
                    elem_classes=["chat-container"],
                )

                # Row 2: Status cards — services, milestones, belief tracker
                gr.Markdown("### Incident status")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=240):
                        gr.Markdown("**Services**")
                        service_display = gr.HTML()
                    with gr.Column(scale=1, min_width=240):
                        gr.Markdown("**Milestones hit**")
                        milestone_display = gr.HTML()
                    with gr.Column(scale=1, min_width=240):
                        gr.Markdown("**🧠 Theory of Mind tracker**")
                        belief_display = gr.HTML()

                # Row 3: Plots — communication flow + reward progress
                gr.Markdown("### Training signal")
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1):
                        comm_flow = gr.Plot(label="Communication flow")
                    with gr.Column(scale=1):
                        reward_plot = gr.Plot(label="Reward progress")

                # Row 4: Timeline (wide, single plot)
                comm_timeline = gr.Plot(label="Communication timeline")

                # Deep-dive: brain scanner, reward inspector, playback (hidden by default)
                with gr.Accordion("🔍 Deep-dive: brain scanner, reward inspector, playback", open=False):
                    gr.Markdown("#### 💭 Agent Brain Scanner")
                    thought_display = gr.HTML(elem_classes=["chat-container"])
                    gr.Markdown("#### Reward inspector")
                    reward_inspector = gr.HTML()
                    gr.Markdown("#### Playback trace")
                    playback_trace = gr.HTML()

                start_btn.click(
                    start_episode,
                    inputs=[task_dropdown, seed_input, agent_mode_toggle, model_name_input, api_base_url_input],
                    outputs=[chat_display, service_display, reward_plot, status_text, milestone_display, comm_flow, comm_timeline, belief_display, thought_display, reward_inspector, playback_trace],
                )
                next_btn.click(
                    next_round,
                    inputs=[task_dropdown],
                    outputs=[chat_display, service_display, reward_plot, status_text, milestone_display, comm_flow, comm_timeline, belief_display, thought_display, reward_inspector, playback_trace],
                )
                auto_btn.click(
                    auto_play,
                    inputs=[task_dropdown, seed_input, agent_mode_toggle, model_name_input, api_base_url_input],
                    outputs=[chat_display, service_display, reward_plot, status_text, milestone_display, comm_flow, comm_timeline, belief_display, thought_display, reward_inspector, playback_trace],
                )
                chaos_btn.click(
                    inject_chaos,
                    outputs=[chat_display, service_display, reward_plot, status_text, milestone_display, comm_flow, comm_timeline, belief_display, thought_display, reward_inspector, playback_trace],
                )
                judge_btn.click(
                    send_judge_message,
                    inputs=[judge_input, judge_target],
                    outputs=[chat_display, service_display, reward_plot, status_text, milestone_display, comm_flow, comm_timeline, belief_display, thought_display, reward_inspector, playback_trace, judge_input],
                )

            # ---- Tab 2: Training Curves ----
            with gr.Tab("📈 Training Curves"):
                gr.Markdown("### Training Progress Visualization")

                with gr.Row():
                    episodes_input = gr.Number(value=30, label="Episodes", precision=0)
                    train_btn = gr.Button("🏋️ Run Training Simulation", variant="primary")

                train_summary = gr.Textbox(label="Summary", interactive=False)

                with gr.Row():
                    reward_curve = gr.Plot(label="Reward Curve")
                    rounds_curve = gr.Plot(label="Rounds to Resolve")

                with gr.Row():
                    milestones_chart = gr.Plot(label="Milestones Per Episode")
                    task_chart = gr.Plot(label="Avg Reward by Task")

                load_btn = gr.Button("📂 Load Existing Metrics")

                train_btn.click(
                    run_training_sim,
                    inputs=[episodes_input],
                    outputs=[reward_curve, rounds_curve, milestones_chart, task_chart, train_summary],
                )
                load_btn.click(
                    load_training_metrics,
                    outputs=[reward_curve, rounds_curve, milestones_chart, task_chart, train_summary],
                )

            # ---- Tab 3: Info ----
            with gr.Tab("📊 Environment Info"):
                gr.Markdown("""
## Architecture

Three specialized agents with **partial observability** cooperate through a **shared communication channel**:

| Agent | Sees | Can Do |
|---|---|---|
| 🚨 **Triage** | Dashboard, alerts, health metrics | Check status, escalate, send messages |
| 🔎 **Diagnosis** | Log files, process table | Read logs, inspect system, send findings |
| 🛠️ **Remediation** | Service status, config files | Restart services, edit configs, verify fixes |

## Tasks

| Task | Difficulty | Rounds | Description |
|---|---|---|---|
| Task 1 | 🟢 Easy | 10 | Coordinated nginx restart |
| Task 2 | 🟡 Medium | 15 | Memory leak + CPU red herring |
| Task 3 | 🔴 Hard | 20 | Cascading failure + phantom alerts (theory-of-mind) |
| Task 4 | ⚫ Expert | 25 | Two simultaneous incidents |

## Reward Design

- **Milestone-based partial credit** — dense signal every round
- **Communication quality scoring** — useful messages +0.05, incorrect info -0.02
- **Time pressure** — -0.01 per round (MTTR optimization)
- **Fatal actions** — kill healthy database = instant game over
- **Phantom alerts** — stale metrics test theory-of-mind reasoning
- **Adaptive difficulty** — environment gets harder as agents improve

## Theory of Mind (Task 3)

Task 3 injects **phantom alerts** — stale cached metrics that appear on Triage's dashboard but don't reflect reality. The Diagnosis agent must:
1. Investigate the phantom lead (Redis)
2. Find nothing wrong
3. Push back on Triage: "Those metrics are stale"
4. Redirect to the real root cause (DB auth failure)

This tests whether agents can model **false beliefs** held by other agents.
                """)

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
