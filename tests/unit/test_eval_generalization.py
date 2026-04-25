"""Tests for round2.war_room.eval_generalization."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from round2.war_room.environment import WarRoomEnvironment
from round2.war_room.models import MultiAgentAction


def test_module_imports_cleanly():
    """Module must import without errors."""
    import round2.war_room.eval_generalization as eg

    # Basic sanity on top-level names we rely on elsewhere.
    assert hasattr(eg, "baseline_policy")
    assert hasattr(eg, "reactive_policy")
    assert hasattr(eg, "run_episode")
    assert hasattr(eg, "evaluate")
    assert hasattr(eg, "render_plot")
    assert eg.DIFFICULTIES == [
        "procedural_easy",
        "procedural_medium",
        "procedural_hard",
    ]


def test_reactive_policy_runs_two_rounds_on_medium():
    """Running the reactive policy for 2 rounds returns valid MultiAgentActions."""
    from round2.war_room.eval_generalization import reactive_policy

    env = WarRoomEnvironment()
    obs = env.reset(task_id="procedural_medium", seed=7)

    for round_num in range(2):
        if obs.done:
            break
        action = reactive_policy(env, obs, round_num)
        assert isinstance(action, MultiAgentAction)
        # Every sub-action must at least be an AgentAction-shaped object.
        assert hasattr(action.triage, "command")
        assert hasattr(action.diagnosis, "command")
        assert hasattr(action.remediation, "command")
        obs = env.step(action)

    # After 2 rounds we should still be in a valid observation shape.
    assert obs is not None


def test_baseline_policy_returns_noop_action():
    """Baseline policy returns a MultiAgentAction with empty commands."""
    from round2.war_room.eval_generalization import baseline_policy

    env = WarRoomEnvironment()
    obs = env.reset(task_id="procedural_easy", seed=0)
    action = baseline_policy(env, obs, 0)
    assert isinstance(action, MultiAgentAction)
    assert action.triage.command == ""
    assert action.diagnosis.command == ""
    assert action.remediation.command == ""


def test_evaluate_writes_expected_json_structure(tmp_path: Path):
    """evaluate() + JSON write produces the expected schema."""
    import round2.war_room.eval_generalization as eg

    results = eg.evaluate(n_seeds=2)

    assert "by_difficulty" in results
    assert results["n_seeds"] == 2

    expected_keys = {
        "scores", "rounds_used", "milestones",
        "resolved_rate", "avg_score",
    }
    for difficulty in eg.DIFFICULTIES:
        assert difficulty in results["by_difficulty"]
        per_diff = results["by_difficulty"][difficulty]
        assert "baseline" in per_diff
        assert "reactive" in per_diff
        for policy_name in ("baseline", "reactive"):
            p = per_diff[policy_name]
            missing = expected_keys - set(p.keys())
            assert not missing, f"{difficulty}/{policy_name} missing keys: {missing}"
            assert len(p["scores"]) == 2
            assert len(p["rounds_used"]) == 2
            assert len(p["milestones"]) == 2
            assert 0.0 <= p["resolved_rate"] <= 1.0
            assert 0.0 <= p["avg_score"] <= 1.0

    # Round-trip through JSON to confirm it's serialisable.
    out_path = tmp_path / "generalization.json"
    out_path.write_text(json.dumps(results, indent=2))
    reloaded = json.loads(out_path.read_text())
    assert reloaded["by_difficulty"].keys() == results["by_difficulty"].keys()


def test_render_plot_writes_png(tmp_path: Path):
    """render_plot() produces a non-empty PNG file."""
    import round2.war_room.eval_generalization as eg

    results = eg.evaluate(n_seeds=2)
    png_path = tmp_path / "generalization.png"
    eg.render_plot(results, str(png_path))

    assert png_path.exists()
    assert png_path.stat().st_size > 1000  # non-trivial PNG
