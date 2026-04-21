"""Per-role observation builders for the Multi-Agent Incident War Room."""

from sre_env.server.simulated_system import SimulatedSystem
from round2.war_room.models import Alert, Message
from round2.war_room.communication import CommunicationChannel


def build_triage_observation(
    system: SimulatedSystem,
    alerts: list[Alert],
    channel: CommunicationChannel,
    round_num: int,
    max_rounds: int,
    agent_name: str = "triage",
) -> str:
    """Build Triage Agent's observation text."""
    lines = [f"[Round {round_num}/{max_rounds}] Role: TRIAGE AGENT"]
    lines.append("─" * 50)

    # Dashboard: service status summary
    lines.append("\nService Status:")
    for name, svc in sorted(system.service_registry.services.items()):
        icon = "●" if svc.status == "running" else "✗"
        lines.append(f"  {icon} {name}: {svc.status} (port {svc.port})")

    # Active alerts
    if alerts:
        lines.append("\nActive Alerts:")
        for a in alerts:
            icon = "🔴" if a.severity == "critical" else "🟡"
            lines.append(f"  {icon} [{a.severity.upper()}] {a.service}: {a.description}")
    else:
        lines.append("\nNo active alerts.")

    # Health metrics
    procs = system.process_table.processes
    total_cpu = sum(p.cpu_percent for p in procs.values())
    total_mem = sum(p.memory_mb for p in procs.values())
    running = sum(1 for s in system.service_registry.services.values() if s.status == "running")
    total = len(system.service_registry.services)
    lines.append(f"\nHealth: {running}/{total} services healthy | CPU: {total_cpu:.1f}% | Memory: {total_mem:.0f}MB/8192MB")

    # Messages
    msgs = channel.get_messages_for(agent_name, since_round=max(0, round_num - 1))
    if msgs:
        lines.append("\nMessages:")
        for m in msgs:
            lines.append(f"  [{m.from_agent}→{m.to_agent}] {m.content}")

    lines.append("\nAvailable commands: get_dashboard, get_alerts, get_health_summary, escalate <agent> <description>, send_message <to> <content>")

    return "\n".join(lines)


def build_diagnosis_observation(
    system: SimulatedSystem,
    channel: CommunicationChannel,
    round_num: int,
    max_rounds: int,
    prev_output: str = "",
    agent_name: str = "diagnosis",
) -> str:
    """Build Diagnosis Agent's observation text."""
    lines = [f"[Round {round_num}/{max_rounds}] Role: DIAGNOSIS AGENT"]
    lines.append("─" * 50)

    if prev_output:
        lines.append(f"\nCommand output:\n{prev_output}")

    # Messages
    msgs = channel.get_messages_for(agent_name, since_round=max(0, round_num - 1))
    if msgs:
        lines.append("\nMessages:")
        for m in msgs:
            lines.append(f"  [{m.from_agent}→{m.to_agent}] {m.content}")

    lines.append("\nAvailable commands: cat, grep, tail, ps, top, journalctl, dmesg, send_message <to> <content>")

    return "\n".join(lines)


def build_remediation_observation(
    system: SimulatedSystem,
    channel: CommunicationChannel,
    round_num: int,
    max_rounds: int,
    prev_output: str = "",
    agent_name: str = "remediation",
) -> str:
    """Build Remediation Agent's observation text."""
    lines = [f"[Round {round_num}/{max_rounds}] Role: REMEDIATION AGENT"]
    lines.append("─" * 50)

    # Service statuses (remediation can see these)
    lines.append("\nService Status:")
    for name, svc in sorted(system.service_registry.services.items()):
        icon = "●" if svc.status == "running" else "✗"
        deps = f" (deps: {', '.join(svc.dependencies)})" if svc.dependencies else ""
        lines.append(f"  {icon} {name}: {svc.status}{deps}")

    if prev_output:
        lines.append(f"\nCommand output:\n{prev_output}")

    # Messages
    msgs = channel.get_messages_for(agent_name, since_round=max(0, round_num - 1))
    if msgs:
        lines.append("\nMessages:")
        for m in msgs:
            lines.append(f"  [{m.from_agent}→{m.to_agent}] {m.content}")

    lines.append("\nAvailable commands: systemctl restart/stop <svc>, edit <path> <old> <new>, kill -9 <PID>, curl <url>, cat <config_path>, send_message <to> <content>")

    return "\n".join(lines)
