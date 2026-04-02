"""Unit tests for CommandParser — tasks 3.1 through 3.4."""

from datetime import datetime

import pytest

from sre_env.server.command_parser import CommandParser
from sre_env.server.models import ParsedCommand
from sre_env.server.simulated_system import SimulatedSystem


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _make_system() -> SimulatedSystem:
    """Build a minimal SimulatedSystem for testing."""
    system = SimulatedSystem()

    # Filesystem
    system.filesystem.write_file("/var/log/app.log", "line1\nline2\nline3\nline4\nline5\n")
    system.filesystem.write_file("/etc/app/config.yml", "password: wrong_pass\nhost: localhost\n")
    system.filesystem.mkdir_p("/tmp")

    # Services
    from sre_env.server.models import ServiceRecord
    system.service_registry.services["nginx"] = ServiceRecord(
        name="nginx", status="running", port=80, pid=1000,
        started_at=datetime(2024, 1, 15, 9, 0, 0),
    )
    system.service_registry.services["app_server"] = ServiceRecord(
        name="app_server", status="crashed", port=8080,
        dependencies=["nginx"],
    )
    system.service_registry.services["db"] = ServiceRecord(
        name="db", status="running", port=5432, pid=1001,
        started_at=datetime(2024, 1, 15, 9, 0, 0),
    )

    # Processes
    system.process_table.add_process("nginx", 2.0, 128.0, "running", "nginx")
    system.process_table.add_process("postgres", 5.0, 256.0, "running", "db")
    system.process_table.add_process("worker", 10.0, 512.0, "running")

    # Logs
    system.log_buffer.append(
        datetime(2024, 1, 15, 9, 55, 0), "ERROR", "nginx", "segfault"
    )
    system.log_buffer.append(
        datetime(2024, 1, 15, 9, 56, 0), "INFO", "app_server", "started"
    )

    return system


@pytest.fixture
def parser():
    return CommandParser()


@pytest.fixture
def system():
    return _make_system()


# =================================================================
# Task 3.1 — parse() and format_command()
# =================================================================

class TestParse:
    def test_simple_command(self, parser):
        p = parser.parse("cat /var/log/app.log")
        assert p.command == "cat"
        assert p.args == ["/var/log/app.log"]
        assert p.flags == {}

    def test_command_with_flag(self, parser):
        p = parser.parse("tail -n 5 /var/log/app.log")
        assert p.command == "tail"
        assert p.flags["-n"] == "5"
        assert p.args == ["/var/log/app.log"]

    def test_command_with_boolean_flag(self, parser):
        p = parser.parse("kill -9 1234")
        assert p.command == "kill"
        assert "-9" in p.flags
        assert p.args == ["1234"]

    def test_grep_with_pattern(self, parser):
        p = parser.parse("grep error /var/log/app.log")
        assert p.command == "grep"
        assert p.args == ["error", "/var/log/app.log"]

    def test_echo_preserves_args(self, parser):
        p = parser.parse('echo "hello world"')
        assert p.command == "echo"
        assert p.args == ["hello world"]

    def test_empty_string(self, parser):
        p = parser.parse("")
        assert p.command == ""

    def test_systemctl_restart(self, parser):
        p = parser.parse("systemctl restart nginx")
        assert p.command == "systemctl"
        assert p.args == ["restart", "nginx"]

    def test_raw_preserved(self, parser):
        raw = "cat /etc/config"
        p = parser.parse(raw)
        assert p.raw == raw

    def test_edit_with_quoted_args(self, parser):
        p = parser.parse('edit /etc/app/config.yml "old text" "new text"')
        assert p.command == "edit"
        assert p.args == ["/etc/app/config.yml", "old text", "new text"]


class TestFormatCommand:
    def test_round_trip_simple(self, parser):
        raw = "cat /var/log/app.log"
        p = parser.parse(raw)
        formatted = parser.format_command(p)
        p2 = parser.parse(formatted)
        assert p.command == p2.command
        assert p.args == p2.args
        assert p.flags == p2.flags

    def test_round_trip_with_flags(self, parser):
        raw = "tail -n 5 /var/log/app.log"
        p = parser.parse(raw)
        formatted = parser.format_command(p)
        p2 = parser.parse(formatted)
        assert p.command == p2.command
        assert p.args == p2.args
        assert p.flags == p2.flags

    def test_format_includes_command(self, parser):
        p = ParsedCommand(command="ls", args=["/tmp"], flags={})
        assert parser.format_command(p).startswith("ls")


# =================================================================
# Task 3.2 — Read-only commands
# =================================================================

class TestReadOnlyCommands:
    def test_cat(self, parser, system):
        out = parser.execute("cat /var/log/app.log", system)
        assert "line1" in out
        assert "line5" in out

    def test_grep(self, parser, system):
        out = parser.execute("grep line3 /var/log/app.log", system)
        assert "line3" in out
        assert "line1" not in out

    def test_tail_default(self, parser, system):
        out = parser.execute("tail /var/log/app.log", system)
        # File has 5 non-empty lines + trailing empty, default 10 returns all
        assert "line1" in out

    def test_tail_n(self, parser, system):
        out = parser.execute("tail -n 2 /var/log/app.log", system)
        assert "line5" in out
        # line1 should not be in last 2 lines
        assert "line1" not in out

    def test_head_default(self, parser, system):
        out = parser.execute("head /var/log/app.log", system)
        assert "line1" in out

    def test_head_n(self, parser, system):
        out = parser.execute("head -n 2 /var/log/app.log", system)
        assert "line1" in out
        assert "line2" in out
        assert "line3" not in out

    def test_ls_root(self, parser, system):
        out = parser.execute("ls /", system)
        assert "var" in out
        assert "etc" in out
        assert "tmp" in out

    def test_ps_aux(self, parser, system):
        out = parser.execute("ps aux", system)
        assert "PID" in out
        assert "nginx" in out
        assert "postgres" in out

    def test_top(self, parser, system):
        out = parser.execute("top", system)
        assert "CPU" in out
        assert "MEM" in out
        assert "PID" in out

    def test_df(self, parser, system):
        out = parser.execute("df", system)
        assert "/dev/sda1" in out
        assert "100G" in out

    def test_free(self, parser, system):
        out = parser.execute("free", system)
        assert "Mem:" in out
        assert "8192" in out

    def test_netstat(self, parser, system):
        out = parser.execute("netstat", system)
        # nginx is running on port 80
        assert "80" in out
        assert "nginx" in out
        # app_server is crashed, should NOT appear
        assert "8080" not in out

    def test_echo(self, parser, system):
        out = parser.execute("echo hello world", system)
        assert out == "hello world"

    def test_help(self, parser, system):
        out = parser.execute("help", system)
        assert "cat" in out
        assert "grep" in out
        assert "kill" in out


# =================================================================
# Task 3.3 — Mutating commands
# =================================================================

class TestMutatingCommands:
    def test_kill_process(self, parser, system):
        # Get a PID
        pids = list(system.process_table.processes.keys())
        pid = pids[0]
        out = parser.execute(f"kill -9 {pid}", system)
        assert "killed" in out.lower() or "killed" in out
        assert pid not in system.process_table.processes

    def test_kill_without_flag(self, parser, system):
        pids = list(system.process_table.processes.keys())
        pid = pids[0]
        out = parser.execute(f"kill {pid}", system)
        assert "killed" in out.lower()

    def test_systemctl_restart(self, parser, system):
        # app_server depends on nginx which is running
        out = parser.execute("systemctl restart app_server", system)
        svc = system.service_registry.get_service("app_server")
        assert svc.status == "running"
        assert "restart" in out.lower() or "successfully" in out.lower()

    def test_systemctl_status(self, parser, system):
        out = parser.execute("systemctl status nginx", system)
        assert "nginx" in out
        assert "running" in out

    def test_systemctl_stop(self, parser, system):
        out = parser.execute("systemctl stop nginx", system)
        svc = system.service_registry.get_service("nginx")
        assert svc.status == "stopped"
        assert "stopped" in out.lower()

    def test_systemctl_start(self, parser, system):
        # Stop first, then start
        parser.execute("systemctl stop db", system)
        out = parser.execute("systemctl start db", system)
        svc = system.service_registry.get_service("db")
        assert svc.status == "running"

    def test_curl_running_service(self, parser, system):
        out = parser.execute("curl http://localhost:80/health", system)
        assert "200 OK" in out
        assert "healthy" in out

    def test_curl_crashed_service(self, parser, system):
        out = parser.execute("curl http://localhost:8080/health", system)
        assert "Connection refused" in out

    def test_curl_by_service_name(self, parser, system):
        out = parser.execute("curl http://nginx/health", system)
        assert "200 OK" in out

    def test_edit_file(self, parser, system):
        out = parser.execute(
            'edit /etc/app/config.yml "wrong_pass" "correct_pass"', system
        )
        assert "updated" in out.lower() or "success" in out.lower()
        content = system.filesystem.read_file("/etc/app/config.yml")
        assert "correct_pass" in content
        assert "wrong_pass" not in content


# =================================================================
# Task 3.4 — Error handling
# =================================================================

class TestErrorHandling:
    def test_unknown_command(self, parser, system):
        out = parser.execute("foobar", system)
        assert "bash: foobar: command not found" == out

    def test_cat_nonexistent_file(self, parser, system):
        out = parser.execute("cat /no/such/file", system)
        assert "No such file or directory: /no/such/file" == out

    def test_grep_nonexistent_file(self, parser, system):
        out = parser.execute("grep pattern /no/file", system)
        assert "No such file or directory: /no/file" == out

    def test_kill_nonexistent_pid(self, parser, system):
        out = parser.execute("kill -9 99999", system)
        assert "No such process: 99999" == out

    def test_systemctl_nonexistent_service(self, parser, system):
        out = parser.execute("systemctl status nonexistent", system)
        assert "Unit nonexistent not found" == out

    def test_systemctl_restart_nonexistent(self, parser, system):
        out = parser.execute("systemctl restart nonexistent", system)
        assert "Unit nonexistent not found" == out

    def test_edit_pattern_not_found(self, parser, system):
        out = parser.execute(
            'edit /etc/app/config.yml "not_in_file" "replacement"', system
        )
        assert "edit: pattern not found in /etc/app/config.yml" == out

    def test_edit_nonexistent_file(self, parser, system):
        out = parser.execute(
            'edit /no/file "old" "new"', system
        )
        assert "No such file or directory: /no/file" == out

    def test_cat_no_args(self, parser, system):
        out = parser.execute("cat", system)
        assert "invalid usage" in out.lower()

    def test_grep_no_args(self, parser, system):
        out = parser.execute("grep", system)
        assert "invalid usage" in out.lower()

    def test_kill_no_args(self, parser, system):
        out = parser.execute("kill", system)
        assert "invalid usage" in out.lower()

    def test_systemctl_no_args(self, parser, system):
        out = parser.execute("systemctl", system)
        assert "invalid usage" in out.lower()

    def test_execute_never_raises(self, parser, system):
        """Execute should never raise, even with garbage input."""
        # These should all return strings, not raise
        for cmd in ["", "   ", "kill abc", "tail", "head"]:
            result = parser.execute(cmd, system)
            assert isinstance(result, str)
