"""Unit tests for SimulatedSystem aggregate class."""

from datetime import datetime, timedelta

from sre_env.server.simulated_system import SimulatedSystem


def _make_system_with_services() -> SimulatedSystem:
    """Helper: build a SimulatedSystem with a few services and processes."""
    sys = SimulatedSystem()

    # Add services
    from sre_env.server.models import ServiceRecord

    sys.service_registry.services["nginx"] = ServiceRecord(
        name="nginx", status="running", port=80, dependencies=[]
    )
    sys.service_registry.services["app_server"] = ServiceRecord(
        name="app_server", status="running", port=8080, dependencies=["nginx"]
    )
    sys.service_registry.services["postgres"] = ServiceRecord(
        name="postgres", status="running", port=5432, dependencies=[]
    )

    # Add processes linked to services
    pid_nginx = sys.process_table.add_process(
        "nginx", cpu=5.0, mem=128.0, status="running", service_name="nginx"
    )
    pid_app = sys.process_table.add_process(
        "app_server", cpu=10.0, mem=256.0, status="running", service_name="app_server"
    )
    pid_pg = sys.process_table.add_process(
        "postgres", cpu=3.0, mem=512.0, status="running", service_name="postgres"
    )

    # Wire PIDs into service records
    sys.service_registry.services["nginx"].pid = pid_nginx
    sys.service_registry.services["app_server"].pid = pid_app
    sys.service_registry.services["postgres"].pid = pid_pg

    return sys


# ------------------------------------------------------------------
# kill_process tests
# ------------------------------------------------------------------

class TestKillProcess:
    def test_kill_existing_process_linked_to_service(self):
        sys = _make_system_with_services()
        nginx_pid = sys.service_registry.services["nginx"].pid

        result = sys.kill_process(nginx_pid)

        assert "killed" in result.lower()
        # Process removed
        assert nginx_pid not in sys.process_table.processes
        # Service cascaded to crashed
        assert sys.service_registry.services["nginx"].status == "crashed"
        assert sys.service_registry.services["nginx"].pid is None

    def test_kill_nonexistent_pid(self):
        sys = SimulatedSystem()
        result = sys.kill_process(9999)
        assert "No such process: 9999" in result

    def test_kill_generates_log_entries(self):
        sys = _make_system_with_services()
        nginx_pid = sys.service_registry.services["nginx"].pid
        log_count_before = len(sys.log_buffer.entries)

        sys.kill_process(nginx_pid)

        # Should have at least 2 new entries: service crash + kill event
        assert len(sys.log_buffer.entries) >= log_count_before + 2

    def test_kill_process_without_service_link(self):
        sys = SimulatedSystem()
        pid = sys.process_table.add_process(
            "orphan", cpu=1.0, mem=32.0, status="running"
        )
        result = sys.kill_process(pid)
        assert "killed" in result.lower()
        assert pid not in sys.process_table.processes


# ------------------------------------------------------------------
# restart_service tests
# ------------------------------------------------------------------

class TestRestartService:
    def test_restart_nonexistent_service(self):
        sys = SimulatedSystem()
        result = sys.restart_service("ghost")
        assert "Unit ghost not found" in result

    def test_restart_with_unmet_dependencies(self):
        sys = _make_system_with_services()
        # Crash nginx so app_server's dependency is unmet
        sys.service_registry.set_status("nginx", "crashed")

        result = sys.restart_service("app_server")
        assert "dependency" in result.lower()
        assert "nginx" in result

    def test_restart_with_met_dependencies(self):
        sys = _make_system_with_services()
        # Crash app_server, then restart it (nginx is running)
        sys.service_registry.set_status("app_server", "crashed")
        old_pid = sys.service_registry.services["app_server"].pid

        result = sys.restart_service("app_server")

        assert "restarted successfully" in result.lower() or "started successfully" in result.lower()
        svc = sys.service_registry.services["app_server"]
        assert svc.status == "running"
        assert svc.pid is not None
        assert svc.pid != old_pid  # new PID assigned

    def test_restart_creates_new_process(self):
        sys = _make_system_with_services()
        sys.service_registry.set_status("postgres", "crashed")

        result = sys.restart_service("postgres")

        svc = sys.service_registry.services["postgres"]
        assert svc.status == "running"
        proc = sys.process_table.get_by_service("postgres")
        assert proc is not None
        assert proc.status == "running"

    def test_restart_logs_event(self):
        sys = _make_system_with_services()
        sys.service_registry.set_status("postgres", "crashed")
        log_count_before = len(sys.log_buffer.entries)

        sys.restart_service("postgres")

        assert len(sys.log_buffer.entries) > log_count_before

    def test_restart_with_config_validator_failure(self):
        sys = _make_system_with_services()
        sys.service_registry.set_status("postgres", "crashed")

        # Register a validator that always fails
        sys.config_validators["postgres"] = lambda s: False

        result = sys.restart_service("postgres")

        assert "crashed" in result.lower()
        svc = sys.service_registry.services["postgres"]
        assert svc.status == "crashed"
        assert svc.pid is None

    def test_restart_with_config_validator_success(self):
        sys = _make_system_with_services()
        sys.service_registry.set_status("postgres", "crashed")

        # Register a validator that passes
        sys.config_validators["postgres"] = lambda s: True

        result = sys.restart_service("postgres")

        assert "successfully" in result.lower()
        svc = sys.service_registry.services["postgres"]
        assert svc.status == "running"


# ------------------------------------------------------------------
# edit_file tests
# ------------------------------------------------------------------

class TestEditFile:
    def test_edit_existing_file(self):
        sys = SimulatedSystem()
        sys.filesystem.write_file("/etc/app/config.yml", "password: wrong")

        result = sys.edit_file("/etc/app/config.yml", "wrong", "correct")

        assert "updated successfully" in result.lower()
        assert sys.filesystem.read_file("/etc/app/config.yml") == "password: correct"

    def test_edit_pattern_not_found(self):
        sys = SimulatedSystem()
        sys.filesystem.write_file("/etc/app/config.yml", "password: wrong")

        result = sys.edit_file("/etc/app/config.yml", "nonexistent", "new")

        assert "pattern not found" in result.lower()

    def test_edit_nonexistent_file(self):
        sys = SimulatedSystem()
        result = sys.edit_file("/no/such/file", "a", "b")
        assert "No such file" in result


# ------------------------------------------------------------------
# advance_time tests
# ------------------------------------------------------------------

class TestAdvanceTime:
    def test_advances_clock(self):
        sys = SimulatedSystem()
        original = sys.current_time
        sys.advance_time(10)
        assert sys.current_time == original + timedelta(seconds=10)

    def test_generates_metrics_for_running_services(self):
        sys = _make_system_with_services()
        sys.advance_time(5)

        for svc_name in ["nginx", "app_server", "postgres"]:
            latest = sys.metrics_store.latest(svc_name)
            assert latest is not None
            assert latest.timestamp == sys.current_time

    def test_crashed_service_gets_zero_metrics(self):
        sys = _make_system_with_services()
        sys.service_registry.set_status("nginx", "crashed")
        sys.advance_time(5)

        latest = sys.metrics_store.latest("nginx")
        assert latest is not None
        assert latest.cpu_percent == 0.0


# ------------------------------------------------------------------
# snapshot tests
# ------------------------------------------------------------------

class TestSnapshot:
    def test_snapshot_contains_all_components(self):
        sys = _make_system_with_services()
        snap = sys.snapshot()

        assert "process_table" in snap
        assert "filesystem" in snap
        assert "service_registry" in snap
        assert "log_buffer" in snap
        assert "metrics_store" in snap
        assert "current_time" in snap

    def test_snapshot_is_serializable(self):
        import json

        sys = _make_system_with_services()
        snap = sys.snapshot()
        # Should not raise
        json.dumps(snap, default=str)


# ------------------------------------------------------------------
# Cross-component consistency tests
# ------------------------------------------------------------------

class TestCrossComponentConsistency:
    def test_kill_then_restart_gives_new_pid(self):
        sys = _make_system_with_services()
        old_pid = sys.service_registry.services["nginx"].pid

        sys.kill_process(old_pid)
        sys.restart_service("nginx")

        new_pid = sys.service_registry.services["nginx"].pid
        assert new_pid is not None
        assert new_pid != old_pid

    def test_log_timestamps_are_monotonic(self):
        sys = _make_system_with_services()
        nginx_pid = sys.service_registry.services["nginx"].pid

        sys.kill_process(nginx_pid)
        sys.advance_time(5)
        sys.restart_service("nginx")

        timestamps = [e.timestamp for e in sys.log_buffer.entries]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_running_service_has_process(self):
        sys = _make_system_with_services()
        nginx_pid = sys.service_registry.services["nginx"].pid
        sys.kill_process(nginx_pid)
        sys.restart_service("nginx")

        for svc_name, svc in sys.service_registry.services.items():
            if svc.status == "running":
                proc = sys.process_table.get_by_service(svc_name)
                assert proc is not None, f"Running service {svc_name} has no process"
