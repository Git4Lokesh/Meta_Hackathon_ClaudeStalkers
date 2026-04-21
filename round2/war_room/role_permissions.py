"""Role-based command permissions for the Multi-Agent Incident War Room."""

from __future__ import annotations

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "triage": {"get_dashboard", "get_alerts", "get_health_summary", "escalate", "send_message"},
    "diagnosis": {"cat", "grep", "tail", "ps", "top", "journalctl", "dmesg", "send_message"},
    "remediation": {"systemctl", "edit", "kill", "curl", "cat", "send_message"},
}

# Log paths that remediation CANNOT access (they can only read config files)
REMEDIATION_BLOCKED_PATHS = {"/var/log/"}


def validate_command(role: str, command: str) -> tuple[bool, str | None]:
    """Validate whether a command is allowed for the given role.

    Returns (allowed, error_message). error_message is None if allowed.
    """
    if not command.strip():
        return True, None  # Empty command = no-op, allowed

    cmd_base = command.strip().split()[0]

    allowed_commands = ROLE_PERMISSIONS.get(role)
    if allowed_commands is None:
        return False, f"Unknown role: {role}"

    if cmd_base not in allowed_commands:
        return False, f"Error: '{cmd_base}' is not available for the {role} role. Allowed: {', '.join(sorted(allowed_commands))}"

    # Special check: remediation can use 'cat' but only for config files, not logs
    if role == "remediation" and cmd_base == "cat":
        parts = command.strip().split()
        if len(parts) > 1:
            path = parts[1]
            for blocked in REMEDIATION_BLOCKED_PATHS:
                if path.startswith(blocked):
                    return False, f"Error: {role} role cannot access log files at {path}. Use 'send_message' to ask the diagnosis agent."

    return True, None
