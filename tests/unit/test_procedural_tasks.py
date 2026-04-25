"""Unit tests for procedural task generator."""

import pytest

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import MultiAgentAction, AgentAction
from round2.war_room.tasks.procedural import (
    ProceduralTask,
    _SERVICE_CATALOG,
    _CRITICAL_SERVICES,
    _sample_fault,
    _sample_phantom_alerts,
    FaultSpec,
)
import random


class TestProceduralTaskDeterminism:
    """Same (difficulty, seed) should always produce the same scenario."""

    def test_same_seed_same_scenario(self):
        task_a = ProceduralTask(difficulty=0.5)
        task_a.create_initial_state(seed=42)

        task_b = ProceduralTask(difficulty=0.5)
        task_b.create_initial_state(seed=42)

        assert [(f.fault_type, f.target_service) for f in task_a._faults] == \
               [(f.fault_type, f.target_service) for f in task_b._faults]
        assert len(task_a._phantom_alerts) == len(task_b._phantom_alerts)

    def test_different_seeds_different_scenarios(self):
        task_a = ProceduralTask(difficulty=0.5)
        task_a.create_initial_state(seed=42)
        task_b = ProceduralTask(difficulty=0.5)
        task_b.create_initial_state(seed=43)
        # Very high probability (~98%) that different seeds produce different faults
        diff = any(
            (f_a.fault_type, f_a.target_service) != (f_b.fault_type, f_b.target_service)
            for f_a, f_b in zip(task_a._faults, task_b._faults)
        )
        assert diff or len(task_a._faults) != len(task_b._faults)


class TestDifficultyScaling:
    """Difficulty parameter should scale faults, phantoms, and max_rounds."""

    def test_difficulty_zero_minimal_faults(self):
        task = ProceduralTask(difficulty=0.0)
        task.create_initial_state(seed=1)
        assert len(task._faults) == 1
        assert len(task._phantom_alerts) == 0
        assert task.max_rounds == 30

    def test_difficulty_one_maximum_faults(self):
        task = ProceduralTask(difficulty=1.0)
        task.create_initial_state(seed=1)
        assert len(task._faults) == 3
        # Phantoms can be fewer than 4 if the pool overlaps with faulted services
        assert 1 <= len(task._phantom_alerts) <= 4
        assert task.max_rounds == 15

    def test_max_rounds_monotonic(self):
        """Higher difficulty = fewer rounds (tighter time pressure)."""
        t1 = ProceduralTask(difficulty=0.1)
        t2 = ProceduralTask(difficulty=0.5)
        t3 = ProceduralTask(difficulty=0.9)
        assert t1.max_rounds > t2.max_rounds > t3.max_rounds

    def test_difficulty_clamping(self):
        # Out-of-range values should be clamped
        assert ProceduralTask(difficulty=-1.0).difficulty_level == 0.0
        assert ProceduralTask(difficulty=2.0).difficulty_level == 1.0


class TestFaultApplication:
    """Each fault type should produce the expected system state change."""

    def test_crash_creates_crashed_service(self):
        task = ProceduralTask(difficulty=0.0)
        system = task.create_initial_state(seed=1)
        # Should have at least one fault
        assert len(task._faults) == 1
        fault = task._faults[0]
        if fault.fault_type == "crash":
            svc = system.service_registry.services[fault.target_service]
            assert svc.status == "crashed"
            assert svc.pid is None

    def test_memory_leak_adds_worker_process(self):
        # Find a seed that produces a memory_leak fault
        for seed in range(200):
            task = ProceduralTask(difficulty=0.0)
            system = task.create_initial_state(seed=seed)
            if task._faults[0].fault_type == "memory_leak":
                svc_name = task._faults[0].target_service
                # Worker process should exist
                worker_procs = [
                    p for p in system.process_table.processes.values()
                    if p.name == f"{svc_name}_worker"
                ]
                assert len(worker_procs) == 1
                assert worker_procs[0].memory_mb >= 2000
                return
        pytest.skip("No memory_leak fault sampled in 200 seeds")

    def test_auth_failure_writes_config(self):
        for seed in range(500):
            task = ProceduralTask(difficulty=0.0)
            system = task.create_initial_state(seed=seed)
            if task._faults[0].fault_type == "auth_failure":
                content = system.filesystem.read_file("/etc/app/database.yml")
                assert "wrong_password_123" in content
                return
        pytest.skip("No auth_failure fault sampled in 500 seeds")


class TestIntegrationWithEnvironment:
    """ProceduralTask should work end-to-end through WarRoomEnvironment."""

    def test_procedural_via_environment(self):
        env = WarRoomEnvironment()
        obs = env.reset(task_id="procedural", seed=42)
        assert obs.done is False
        assert obs.triage.text
        assert obs.metadata["task_id"] == "procedural"

    def test_procedural_step_produces_reward(self):
        env = WarRoomEnvironment()
        env.reset(task_id="procedural", seed=42)
        action = MultiAgentAction(
            triage=AgentAction(command="get_dashboard"),
            diagnosis=AgentAction(command=""),
            remediation=AgentAction(command=""),
        )
        obs = env.step(action)
        assert isinstance(obs.team_reward, float)
        assert 0.0 <= obs.team_reward <= 1.0

    @pytest.mark.parametrize("alias", [
        "procedural_easy", "procedural_medium", "procedural_hard",
    ])
    def test_difficulty_aliases_work(self, alias):
        env = WarRoomEnvironment()
        obs = env.reset(task_id=alias, seed=7)
        assert obs.done is False

    def test_procedural_many_seeds(self):
        """Stress test: 20 different seeds all produce valid episodes."""
        for seed in range(20):
            env = WarRoomEnvironment()
            obs = env.reset(task_id="procedural", seed=seed)
            assert obs.triage.text


class TestFatalCheck:
    """Critical service protection must not trigger on initial crashed state."""

    def test_initial_crashed_service_not_fatal(self):
        # Even if seed produces a scenario where a critical service is faulted,
        # the fatal check should not fire on the initial state.
        env = WarRoomEnvironment()
        env.reset(task_id="procedural", seed=1)
        # Very first step — no agent action has killed anything
        action = MultiAgentAction(
            triage=AgentAction(command="get_dashboard"),
            diagnosis=AgentAction(command="ps aux"),
            remediation=AgentAction(command=""),
        )
        obs = env.step(action)
        # Even if initial state has crashed services, fatal should not trigger
        assert not obs.metadata.get("penalties_applied", []) or \
               "FATAL" not in obs.metadata.get("penalties_applied", [])


class TestFaultSampling:
    """Internal fault sampling functions should respect constraints."""

    def test_sample_fault_respects_already_faulted(self):
        rng = random.Random(42)
        faulted = {"nginx", "app_server"}
        f = _sample_fault(rng, faulted)
        assert f.target_service not in faulted

    def test_sample_phantom_alerts_count(self):
        rng = random.Random(42)
        alerts = _sample_phantom_alerts(rng, n=3, faulted_services=set())
        assert len(alerts) == 3

    def test_sample_phantom_alerts_excludes_faulted(self):
        rng = random.Random(42)
        alerts = _sample_phantom_alerts(rng, n=4, faulted_services={"redis", "monitoring"})
        for alert in alerts:
            assert alert.service not in {"redis", "monitoring"}

    def test_critical_services_not_crash_targets(self):
        rng = random.Random(42)
        # Sample many crash faults — none should target a critical service
        for _ in range(50):
            f = _sample_fault(rng, set(), allowed_types=["crash"])
            assert f.target_service not in _CRITICAL_SERVICES


class TestDiskFullFault:
    """The disk_full primitive should: be sampleable, apply to the filesystem,
    and have a milestone that fires when the bloated log gets cleared."""

    def test_disk_full_is_sampleable(self):
        rng = random.Random(0)
        types_seen: set[str] = set()
        # Sample enough times that the 5 fault types should all appear.
        for _ in range(200):
            spec = _sample_fault(rng, already_faulted=set())
            types_seen.add(spec.fault_type)
        assert "disk_full" in types_seen

    def test_disk_full_injects_bloated_log_and_syslog_entry(self):
        env = WarRoomEnvironment()
        # Force disk_full by constructing a ProceduralTask and patching its
        # sampler to emit only disk_full faults for this seed.
        task = ProceduralTask(difficulty=0.5)
        spec = FaultSpec(
            fault_type="disk_full",
            target_service="nginx",
            params={"disk_percent": 98, "log_path": "/var/log/nginx/access.log"},
        )
        task._faults = [spec]
        task._phantom_alerts = []
        system = task._round1_task = None  # ensure no other state
        from round2.war_room.tasks.procedural import _build_base_system, _apply_disk_full
        system = _build_base_system(random.Random(0))
        _apply_disk_full(system, spec)

        # Bloated log file exists and is large.
        content = system.filesystem.read_file("/var/log/nginx/access.log")
        assert len(content) > 500

        # Kernel / service syslog entries present.
        log_messages = [e.message for e in system.log_buffer.entries]
        assert any("No space left" in m for m in log_messages)
        assert any("ENOSPC" in m for m in log_messages)

    def test_disk_full_milestone_fires_when_log_truncated(self):
        from round2.war_room.tasks.procedural import (
            _make_milestones_for_fault,
            _apply_disk_full,
            _build_base_system,
        )
        spec = FaultSpec(
            fault_type="disk_full",
            target_service="nginx",
            params={"disk_percent": 98, "log_path": "/var/log/nginx/access.log"},
        )
        system = _build_base_system(random.Random(0))
        _apply_disk_full(system, spec)

        milestones = _make_milestones_for_fault(spec)
        remediation_m = next(m for m in milestones if m.name.startswith("remediation_clears_"))

        # Before cleanup — milestone should not be satisfied.
        assert remediation_m.check(None, system, {}, None) is False

        # Simulate a log rotation / truncation.
        system.filesystem.write_file(spec.params["log_path"], "")
        assert remediation_m.check(None, system, {}, None) is True
