"""CommandParser: parse and execute Linux-style commands against a SimulatedSystem.

Stateless parser that takes a command string and a SimulatedSystem reference,
executes the command, and returns output text.  Mutating commands (kill,
systemctl restart, edit) modify the system in-place.  The execute method
NEVER raises exceptions — it always returns a string.
"""

from __future__ import annotations

import re
import shlex
from typing import Optional

from sre_env.server.models import ParsedCommand
from sre_env.server.simulated_system import SimulatedSystem


class CommandParser:
    """Parse and execute Linux-style commands."""

    SUPPORTED_COMMANDS = [
        "cat", "grep", "tail", "head", "ls", "ps", "top",
        "kill", "systemctl", "curl", "df", "free", "netstat",
        "edit", "echo", "help", "journalctl", "dmesg", "message",
    ]

    # -----------------------------------------------------------------
    # Parsing
    # -----------------------------------------------------------------

    def parse(self, raw: str) -> ParsedCommand:
        """Parse a raw command string into a ParsedCommand.

        Uses ``shlex.split()`` for proper shell-style tokenisation.
        The first token is the command name.  Tokens starting with ``-``
        are treated as flags; if the next token does not start with ``-``
        and is not the last positional arg it is consumed as the flag
        value.  All other tokens are positional args.
        """
        raw = raw.strip()
        try:
            tokens = shlex.split(raw)
        except ValueError:
            return ParsedCommand(command="", args=[], flags={}, raw=raw)

        if not tokens:
            return ParsedCommand(command="", args=[], flags={}, raw=raw)

        command = tokens[0]
        args: list[str] = []
        flags: dict[str, Optional[str]] = {}

        i = 1
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("-"):
                # Purely numeric flags like -9 (signal numbers) are boolean
                flag_body = token[1:]
                if flag_body.isdigit():
                    flags[token] = None
                    i += 1
                # Check if next token is a value for this flag
                elif i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                    # Known boolean flags that shouldn't consume values
                    if token in ("-i", "-v", "-f", "-r"):
                        flags[token] = None
                        i += 1
                    else:
                        flags[token] = tokens[i + 1]
                        i += 2
                else:
                    flags[token] = None
                    i += 1
            else:
                args.append(token)
                i += 1

        return ParsedCommand(command=command, args=args, flags=flags, raw=raw)

    def format_command(self, parsed: ParsedCommand) -> str:
        """Format a ParsedCommand back to a command string."""
        parts: list[str] = [parsed.command]

        for flag, value in parsed.flags.items():
            parts.append(flag)
            if value is not None:
                parts.append(value)

        for arg in parsed.args:
            # Quote args that contain spaces
            if " " in arg:
                parts.append(shlex.quote(arg))
            else:
                parts.append(arg)

        return " ".join(parts)

    # -----------------------------------------------------------------
    # Execution entry-point
    # -----------------------------------------------------------------

    def execute(self, raw: str, system: SimulatedSystem) -> str:
        """Parse and execute a command against the simulated system.

        Never raises exceptions — always returns a string.
        """
        try:
            parsed = self.parse(raw)

            if not parsed.command:
                return "bash: : command not found"

            if parsed.command not in self.SUPPORTED_COMMANDS:
                return f"bash: {parsed.command}: command not found"

            handler = getattr(self, f"_exec_{parsed.command}", None)
            if handler is None:
                return f"bash: {parsed.command}: command not found"

            return handler(parsed, system)
        except Exception as exc:  # noqa: BLE001 — never propagate
            return f"{parsed.command}: error: {exc}"

    # -----------------------------------------------------------------
    # Read-only command handlers (Task 3.2)
    # -----------------------------------------------------------------

    def _exec_cat(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        if not parsed.args:
            return "cat: invalid usage. Try 'help' for available commands"
        path = parsed.args[0]
        try:
            return system.filesystem.read_file(path)
        except ValueError:
            return f"No such file or directory: {path}"

    def _exec_grep(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        if len(parsed.args) < 2:
            return "grep: invalid usage. Try 'help' for available commands"
        pattern = parsed.args[0]
        path = parsed.args[1]
        try:
            content = system.filesystem.read_file(path)
        except ValueError:
            return f"No such file or directory: {path}"
            
        ignore_case = "-i" in parsed.flags
        invert_match = "-v" in parsed.flags
        
        try:
            regex_flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags=regex_flags)
        except re.error:
            return f"grep: invalid regular expression: '{pattern}'"
            
        lines = content.splitlines()
        if invert_match:
            matches = [line for line in lines if not regex.search(line)]
        else:
            matches = [line for line in lines if regex.search(line)]
        return "\n".join(matches)

    def _exec_tail(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        n = 10
        if "-n" in parsed.flags and parsed.flags["-n"] is not None:
            try:
                n = int(parsed.flags["-n"])
            except ValueError:
                return "tail: invalid usage. Try 'help' for available commands"
        if not parsed.args:
            return "tail: invalid usage. Try 'help' for available commands"
        path = parsed.args[0]
        try:
            content = system.filesystem.read_file(path)
        except ValueError:
            return f"No such file or directory: {path}"
        lines = content.splitlines()
        return "\n".join(lines[-n:])

    def _exec_head(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        n = 10
        if "-n" in parsed.flags and parsed.flags["-n"] is not None:
            try:
                n = int(parsed.flags["-n"])
            except ValueError:
                return "head: invalid usage. Try 'help' for available commands"
        if not parsed.args:
            return "head: invalid usage. Try 'help' for available commands"
        path = parsed.args[0]
        try:
            content = system.filesystem.read_file(path)
        except ValueError:
            return f"No such file or directory: {path}"
        lines = content.splitlines()
        return "\n".join(lines[:n])

    def _exec_ls(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        path = parsed.args[0] if parsed.args else "/"
        try:
            entries = system.filesystem.list_dir(path)
        except ValueError:
            return f"No such file or directory: {path}"
        return "\n".join(entries)

    def _exec_ps(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        procs = system.process_table.processes
        header = f"{'PID':<8}{'NAME':<25}{'CPU%':<8}{'MEM(MB)':<10}{'STATUS':<12}{'SERVICE'}"
        lines = [header]
        for proc in sorted(procs.values(), key=lambda p: p.pid):
            svc = proc.service_name or "-"
            lines.append(
                f"{proc.pid:<8}{proc.name:<25}{proc.cpu_percent:<8.1f}"
                f"{proc.memory_mb:<10.1f}{proc.status:<12}{svc}"
            )
        return "\n".join(lines)

    def _exec_top(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        procs = system.process_table.processes
        total_cpu = sum(p.cpu_percent for p in procs.values())
        total_mem = sum(p.memory_mb for p in procs.values())
        summary = (
            f"top - {system.current_time.strftime('%H:%M:%S')}\n"
            f"Tasks: {len(procs)} total\n"
            f"CPU: {total_cpu:.1f}% used\n"
            f"MEM: {total_mem:.1f} MB used\n"
            f""
        )
        header = f"{'PID':<8}{'NAME':<25}{'CPU%':<8}{'MEM(MB)':<10}{'STATUS':<12}{'SERVICE'}"
        lines = [summary, header]
        for proc in sorted(procs.values(), key=lambda p: p.cpu_percent, reverse=True):
            svc = proc.service_name or "-"
            lines.append(
                f"{proc.pid:<8}{proc.name:<25}{proc.cpu_percent:<8.1f}"
                f"{proc.memory_mb:<10.1f}{proc.status:<12}{svc}"
            )
        return "\n".join(lines)

    def _exec_df(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        return (
            "Filesystem  Size  Used  Avail  Use%  Mounted on\n"
            "/dev/sda1   100G  45G   55G    45%   /"
        )

    def _exec_free(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        total_mem_mb = 8192.0
        used_mb = sum(p.memory_mb for p in system.process_table.processes.values())
        free_mb = max(total_mem_mb - used_mb, 0.0)
        return (
            f"{'':15}{'total':>10}{'used':>10}{'free':>10}\n"
            f"{'Mem:':15}{total_mem_mb:>10.0f}{used_mb:>10.0f}{free_mb:>10.0f}"
        )

    def _exec_netstat(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        header = f"{'Proto':<8}{'Local Address':<25}{'State':<15}{'Service'}"
        lines = [header]
        for svc in sorted(system.service_registry.services.values(), key=lambda s: s.port):
            if svc.status == "running":
                lines.append(
                    f"{'tcp':<8}{f'0.0.0.0:{svc.port}':<25}{'LISTEN':<15}{svc.name}"
                )
        return "\n".join(lines)

    def _exec_echo(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        return " ".join(parsed.args)

    def _exec_journalctl(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        """journalctl [-u service] [-n N] [--since "time"]"""
        service = parsed.flags.get("-u")
        n = 20  # default lines
        if "-n" in parsed.flags and parsed.flags["-n"] is not None:
            try:
                n = int(parsed.flags["-n"])
            except ValueError:
                pass

        if service:
            entries = system.log_buffer.tail(n, source=service)
        else:
            entries = system.log_buffer.tail(n)

        if not entries:
            return "-- No entries --"

        lines = []
        for e in entries:
            lines.append(
                f"{e.timestamp.strftime('%b %d %H:%M:%S')} {e.source}"
                f"[{hash(e.source) % 10000}]: {e.severity} {e.message}"
            )
        return "\n".join(lines)

    def _exec_dmesg(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        """Show kernel-level messages (simulated)."""
        entries = system.log_buffer.query(severity="ERROR") + system.log_buffer.query(severity="FATAL")
        entries.sort(key=lambda e: e.timestamp)

        if "-T" in parsed.flags:
            lines = [
                f"[{e.timestamp.strftime('%a %b %d %H:%M:%S %Y')}] {e.source}: {e.message}"
                for e in entries[-30:]
            ]
        else:
            lines = [
                f"[{(e.timestamp.timestamp() % 100000):.6f}] {e.source}: {e.message}"
                for e in entries[-30:]
            ]

        if not lines:
            return "-- No kernel messages --"
        return "\n".join(lines)

    def _exec_help(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        return (
            "Available commands:\n"
            "  cat <path>              - Display file contents\n"
            "  grep <pattern> <path>   - Search for pattern in file\n"
            "  tail [-n N] <path>      - Display last N lines (default 10)\n"
            "  head [-n N] <path>      - Display first N lines (default 10)\n"
            "  ls [path]               - List directory contents\n"
            "  ps aux                  - Show process table\n"
            "  top                     - Show system summary and processes\n"
            "  kill [-9] <PID>         - Kill a process\n"
            "  systemctl <action> <svc>- Manage services (restart|status|stop|start)\n"
            "  curl <url>              - Make HTTP request to service\n"
            "  df                      - Show disk usage\n"
            "  free                    - Show memory usage\n"
            "  netstat                 - Show listening ports\n"
            "  edit <path> <old> <new> - Edit file content\n"
            "  echo <text>             - Print text\n"
            "  journalctl [-u svc] [-n N] - Show journal logs\n"
            "  dmesg [-T]              - Show kernel messages (errors/fatals)\n"
            "  message <target> <text> - Send a message to another agent\n"
            "  help                    - Show this help message"
        )

    # -----------------------------------------------------------------
    # Mutating command handlers (Task 3.3)
    # -----------------------------------------------------------------

    def _exec_kill(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        # Accept both `kill -9 <PID>` and `kill <PID>`
        if not parsed.args:
            return "kill: invalid usage. Try 'help' for available commands"
        pid_str = parsed.args[0]
        try:
            pid = int(pid_str)
        except ValueError:
            return "kill: invalid usage. Try 'help' for available commands"
        result = system.kill_process(pid)
        return result

    def _exec_systemctl(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        if not parsed.args:
            return "systemctl: invalid usage. Try 'help' for available commands"

        action = parsed.args[0]
        if action not in ("restart", "status", "stop", "start"):
            return "systemctl: invalid usage. Try 'help' for available commands"

        if len(parsed.args) < 2:
            return "systemctl: invalid usage. Try 'help' for available commands"

        service_name = parsed.args[1]

        if action == "status":
            return self._systemctl_status(service_name, system)
        elif action == "restart" or action == "start":
            return system.restart_service(service_name)
        elif action == "stop":
            return self._systemctl_stop(service_name, system)

        return "systemctl: invalid usage. Try 'help' for available commands"

    def _systemctl_status(self, service_name: str, system: SimulatedSystem) -> str:
        svc = system.service_registry.get_service(service_name)
        if svc is None:
            return f"Unit {service_name} not found"

        lines = [
            f"● {svc.name}.service",
            f"   Status: {svc.status}",
            f"   PID: {svc.pid or 'N/A'}",
            f"   Started: {svc.started_at.isoformat() if svc.started_at else 'N/A'}",
            f"",
            f"   Recent logs:",
        ]
        recent = system.log_buffer.tail(5, source=service_name)
        if recent:
            for entry in recent:
                lines.append(
                    f"   {entry.timestamp.strftime('%b %d %H:%M:%S')} "
                    f"{entry.severity} {entry.message}"
                )
        else:
            lines.append("   (no recent log entries)")
        return "\n".join(lines)

    def _systemctl_stop(self, service_name: str, system: SimulatedSystem) -> str:
        svc = system.service_registry.get_service(service_name)
        if svc is None:
            return f"Unit {service_name} not found"

        # Kill the associated process if any
        if svc.pid is not None:
            system.process_table.kill_process(svc.pid)

        system.service_registry.set_status(service_name, "stopped")
        svc.pid = None
        system.log_buffer.append(
            timestamp=system.current_time,
            severity="INFO",
            source=service_name,
            message=f"Service {service_name} stopped by user",
        )
        return f"Service {service_name} stopped"

    def _exec_curl(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        if not parsed.args:
            return "curl: invalid usage. Try 'help' for available commands"

        url = parsed.args[0]

        # Parse URL to find host/port or service name
        # Supported formats:
        #   http://host:port/path
        #   http://service_name/path
        #   http://localhost:port/path
        host, port = self._parse_url(url)

        # Try to find the service by port first, then by name
        target_svc = None
        for svc in system.service_registry.services.values():
            if port is not None and svc.port == port:
                target_svc = svc
                break
            if svc.name == host:
                target_svc = svc
                break

        if target_svc is None:
            # Try matching host as service name
            target_svc = system.service_registry.get_service(host)

        if target_svc is None:
            display_port = port if port is not None else 80
            return (
                f"curl: (7) Failed to connect to {host} port "
                f"{display_port}: Connection refused"
            )

        if target_svc.status == "running":
            return 'HTTP/1.1 200 OK\n\n{"status": "healthy"}'
        else:
            return (
                f"curl: (7) Failed to connect to {host} port "
                f"{target_svc.port}: Connection refused"
            )

    @staticmethod
    def _parse_url(url: str) -> tuple[str, int | None]:
        """Extract (host, port) from a URL string."""
        # Strip protocol
        stripped = url
        if "://" in stripped:
            stripped = stripped.split("://", 1)[1]
        # Strip path
        if "/" in stripped:
            stripped = stripped.split("/", 1)[0]
        # Split host:port
        if ":" in stripped:
            parts = stripped.rsplit(":", 1)
            try:
                return parts[0], int(parts[1])
            except ValueError:
                return stripped, None
        return stripped, None

    def _exec_edit(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        if len(parsed.args) < 3:
            return "edit: invalid usage. Try 'help' for available commands"
        path = parsed.args[0]
        old_content = parsed.args[1]
        new_content = parsed.args[2]
        result = system.edit_file(path, old_content, new_content)
        return result

    def _exec_message(self, parsed: ParsedCommand, system: SimulatedSystem) -> str:
        if len(parsed.args) < 2:
            return "message: invalid usage. Try 'message <target_agent> <text>'"
        
        target = parsed.args[0]
        # Rest of args is the message body
        msg_body = " ".join(parsed.args[1:])
        
        # We append it to a simulated message broker log for the orchestrator to read if needed,
        # but primarily returning the formatted text is used by inference.py's message history.
        return f"[MESSAGE TO {target.upper()}]: {msg_body}"

