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

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import MultiAgentAction, AgentAction, Message

# ---- Global state ----
env = WarRoomEnvironment()
current_obs = None
round_num = 0
chat_history = []
reward_history = []
milestone_list = []

ROLE_ICONS = {"triage": "🚨", "diagnosis": "🔎", "remediation": "🛠️"}
ROLE_COLORS = {"triage": "#FFD700", "diagnosis": "#00CED1", "remediation": "#32CD32"}

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


def _format_chat_entry(role: str, command: str, message_to: str, message_content: str) -> str:
    """Format a chat entry in HTML."""
    icon = ROLE_ICONS.get(role, "❓")
    color = ROLE_COLORS.get(role, "#999")
    parts = []
    if command:
        parts.append(f'<span style="background:#1a1a2e;padding:2px 6px;border-radius:4px;font-family:monospace;color:#0f0">{command}</span>')
    if message_to and message_content:
        parts.append(f'<span style="color:#aaa">→ @{message_to}:</span> <em>{message_content}</em>')
    content = " ".join(parts) if parts else "<em>(no action)</em>"
    return f'<div style="margin:4px 0;padding:8px;border-left:3px solid {color};background:#0d1117"><strong style="color:{color}">{icon} @{role.capitalize()}</strong> {content}</div>'


def _service_status_html(system_snapshot: dict) -> str:
    """Build HTML for service status panel."""
    if not system_snapshot:
        return "<em>No data</em>"
    services = system_snapshot.get("service_registry", {}).get("services", {})
    rows = []
    for name, svc in sorted(services.items()):
        status = svc.get("status", "unknown")
        icon = "🟢" if status == "running" else "🔴" if status == "crashed" else "🟡"
        rows.append(f"<tr><td>{icon}</td><td><strong>{name}</strong></td><td>{status}</td><td>{svc.get('port', '?')}</td></tr>")
    return f'<table style="width:100%"><tr><th></th><th>Service</th><th>Status</th><th>Port</th></tr>{"".join(rows)}</table>'


def _milestone_html(milestones: list) -> str:
    """Build HTML checklist for milestones."""
    if not milestones:
        return "<em>No milestones yet</em>"
    items = [f"<li>✅ {m}</li>" for m in milestones]
    return f"<ul>{''.join(items)}</ul>"


def _reward_plot(rewards: list) -> plt.Figure:
    """Create a reward progress plot."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(1, len(rewards) + 1), rewards, 'b-o', markersize=6, linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Team Reward")
    ax.set_title("Reward Progress")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(1, len(rewards) + 1), rewards, alpha=0.2)
    plt.tight_layout()
    return fig


# ---- Tab 1: War Room ----

def start_episode(task_id: str, seed: int):
    global env, current_obs, round_num, chat_history, reward_history, milestone_list
    env = WarRoomEnvironment()
    current_obs = env.reset(task_id=task_id, seed=seed)
    round_num = 0
    chat_history = []
    reward_history = []
    milestone_list = []

    task_name = current_obs.metadata.get("task_name", task_id)
    difficulty = current_obs.metadata.get("difficulty", "?")
    max_rounds = current_obs.metadata.get("max_rounds", "?")

    header = f'<div style="background:#1a1a2e;padding:12px;border-radius:8px;margin-bottom:8px"><h3 style="color:#ff4444;margin:0">🔧 INCIDENT WAR ROOM</h3><p style="color:#aaa;margin:4px 0">{task_name} ({difficulty}) — Max {max_rounds} rounds</p></div>'
    chat_history.append(header)

    system_html = _service_status_html(env.state.simulated_system)
    chat_html = "\n".join(chat_history)

    fig = _reward_plot([0.0])

    return chat_html, system_html, fig, "Episode started. Click 'Next Round' to step.", _milestone_html([])


def next_round(task_id: str):
    global current_obs, round_num, chat_history, reward_history, milestone_list

    if current_obs is None:
        return "\n".join(chat_history), "<em>Start an episode first</em>", _reward_plot([0]), "Start an episode first.", _milestone_html([])

    if current_obs.done:
        return "\n".join(chat_history), _service_status_html(env.state.simulated_system), _reward_plot(reward_history), "Episode complete!", _milestone_html(milestone_list)

    steps = HEURISTIC_STEPS.get(task_id, HEURISTIC_STEPS["task1"])

    if round_num >= len(steps):
        # No more heuristic steps — send no-ops
        action = MultiAgentAction()
    else:
        step_data = steps[round_num]
        action = _build_action(step_data, round_num + 1)

    current_obs = env.step(action)
    round_num += 1
    reward_history.append(current_obs.team_reward)

    # Add round header
    chat_history.append(f'<div style="color:#666;margin:8px 0;border-top:1px solid #333;padding-top:4px"><strong>Round {round_num}</strong></div>')

    # Add agent actions to chat
    for role in ["triage", "diagnosis", "remediation"]:
        a = getattr(action, role)
        if a.command or a.message:
            msg_to = a.message.to_agent if a.message else None
            msg_content = a.message.content if a.message else None
            chat_history.append(_format_chat_entry(role, a.command, msg_to, msg_content))

    # Update milestones
    if current_obs.done and "milestones_achieved" in current_obs.metadata:
        milestone_list = current_obs.metadata["milestones_achieved"]
    elif hasattr(env, '_grader') and env._grader:
        milestone_list = sorted(env._grader.achieved)

    # Status message
    status = f"Round {round_num} | Reward: {current_obs.team_reward:.3f}"
    if current_obs.done:
        score = current_obs.metadata.get("score", current_obs.team_reward)
        chat_history.append(f'<div style="background:#0a3d0a;padding:12px;border-radius:8px;margin-top:8px"><h3 style="color:#0f0;margin:0">✅ INCIDENT RESOLVED</h3><p style="color:#aaa">Score: {score:.3f} | Rounds: {round_num} | Milestones: {len(milestone_list)}</p></div>')
        status = f"✅ RESOLVED — Score: {score:.3f}"

    chat_html = "\n".join(chat_history)
    system_html = _service_status_html(env.state.simulated_system)
    fig = _reward_plot(reward_history)

    return chat_html, system_html, fig, status, _milestone_html(milestone_list)


def auto_play(task_id: str, seed: int):
    """Run entire episode automatically."""
    result = start_episode(task_id, seed)

    for _ in range(30):  # Max iterations
        if current_obs and current_obs.done:
            break
        result = next_round(task_id)

    return result


# ---- Tab 2: Training Curves ----

def load_training_metrics():
    """Load and plot training metrics."""
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
    ax1.plot(episodes, rewards, 'b-o', markersize=3, linewidth=1, alpha=0.7)
    if len(rewards) >= 5:
        rolling = [sum(rewards[max(0, i - 4):i + 1]) / min(i + 1, 5) for i in range(len(rewards))]
        ax1.plot(episodes, rolling, 'r-', linewidth=2, label='5-ep rolling avg')
        ax1.legend()
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Team Reward")
    ax1.set_title("Reward Over Training Episodes")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 2: Rounds to resolve
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(episodes, rounds_used, 'g-o', markersize=3, linewidth=1)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rounds Used")
    ax2.set_title("Rounds to Resolve (↓ = more efficient)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 3: Milestones
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.bar(episodes, milestones, color='purple', alpha=0.7)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Milestones")
    ax3.set_title("Milestones Achieved Per Episode")
    ax3.grid(True, alpha=0.3)
    plt.tight_layout()

    # Plot 4: By task
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    task_rewards = {}
    for t, r in zip(tasks, rewards):
        task_rewards.setdefault(t, []).append(r)
    for tid, tr in sorted(task_rewards.items()):
        ax4.bar(tid, sum(tr) / len(tr), alpha=0.7)
    ax4.set_xlabel("Task")
    ax4.set_ylabel("Avg Reward")
    ax4.set_title("Average Reward by Task")
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
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

APP_THEME = gr.themes.Base(primary_hue="red", neutral_hue="slate")
APP_CSS = """
.chat-panel { max-height: 500px; overflow-y: auto; background: #0d1117; padding: 12px; border-radius: 8px; }
.status-panel { background: #0d1117; padding: 12px; border-radius: 8px; }
"""


def build_app():
    with gr.Blocks(title="Multi-Agent Incident War Room") as app:
        gr.Markdown("# 🔧 Multi-Agent Incident War Room\n**Three AI agents cooperate to diagnose and fix production incidents.**")

        with gr.Tabs():
            # ---- Tab 1: War Room ----
            with gr.Tab("🔧 War Room"):
                with gr.Row():
                    task_dropdown = gr.Dropdown(
                        choices=["task1", "task2", "task3", "task4"],
                        value="task1",
                        label="Task",
                    )
                    seed_input = gr.Number(value=42, label="Seed", precision=0)

                with gr.Row():
                    start_btn = gr.Button("▶️ Start Episode", variant="primary")
                    next_btn = gr.Button("⏭️ Next Round")
                    auto_btn = gr.Button("⏩ Auto-Play", variant="secondary")

                status_text = gr.Textbox(label="Status", interactive=False)

                with gr.Row():
                    with gr.Column(scale=3):
                        chat_display = gr.HTML(label="Agent Chat", elem_classes=["chat-panel"])
                    with gr.Column(scale=1):
                        service_display = gr.HTML(label="Services", elem_classes=["status-panel"])
                        milestone_display = gr.HTML(label="Milestones")

                reward_plot = gr.Plot(label="Reward Progress")

                start_btn.click(
                    start_episode,
                    inputs=[task_dropdown, seed_input],
                    outputs=[chat_display, service_display, reward_plot, status_text, milestone_display],
                )
                next_btn.click(
                    next_round,
                    inputs=[task_dropdown],
                    outputs=[chat_display, service_display, reward_plot, status_text, milestone_display],
                )
                auto_btn.click(
                    auto_play,
                    inputs=[task_dropdown, seed_input],
                    outputs=[chat_display, service_display, reward_plot, status_text, milestone_display],
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
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=APP_THEME, css=APP_CSS)
