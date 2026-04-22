"""Belief State Tracker for Theory-of-Mind Visualization.

Tracks what each agent **believes** to be true based on:
  - Their direct observations (what they've seen firsthand)
  - Messages received (secondhand info from other agents)
  - Commands they've run

Compares beliefs against **ground truth** (actual system state) to detect:
  - ✅ Correct beliefs
  - ❌ False beliefs (e.g., trusting a phantom alert)
  - ⚠️ Belief conflicts (two agents disagree)
  - 🧠 Theory-of-Mind events (agent detects another's false belief)

Usage:
    tracker = BeliefStateTracker(ground_truth_services={"nginx": "crashed", "redis": "running"})
    tracker.record_observation("triage", "dashboard", {"nginx": "crashed", "redis": "high_memory"})
    tracker.record_message("triage", "diagnosis", "Redis memory is critical!")
    tracker.record_pushback("diagnosis", "triage", "redis", "Redis is fine, metrics are stale")
    snapshot = tracker.get_snapshot()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Belief:
    """A single belief held by an agent about a service/entity."""
    entity: str              # e.g., "nginx", "redis", "db_connector"
    attribute: str           # e.g., "status", "memory", "cpu"
    believed_value: str      # What the agent thinks is true
    source: str              # "observation", "message", "log", "dashboard"
    source_agent: str        # Who provided this info ("self", "triage", etc.)
    round_acquired: int      # When the agent learned this
    correct: Optional[bool] = None  # True/False vs ground truth, None if unknown


@dataclass
class BeliefConflict:
    """A detected conflict where two agents disagree."""
    entity: str
    attribute: str
    agent_a: str
    agent_a_value: str
    agent_b: str
    agent_b_value: str
    ground_truth: Optional[str] = None
    resolved: bool = False
    resolved_by: Optional[str] = None
    round_detected: int = 0


@dataclass
class TheoryOfMindEvent:
    """A theory-of-mind event: one agent detects another's false belief."""
    detector_agent: str      # Agent that detected the false belief
    target_agent: str        # Agent holding the false belief
    entity: str              # What the false belief is about
    false_value: str         # What the target agent wrongly believes
    true_value: str          # What the detector agent found is actually true
    round_number: int
    pushback_message: str = ""  # The actual pushback message


class BeliefStateTracker:
    """Tracks and compares agent beliefs against ground truth."""

    def __init__(
        self,
        ground_truth: dict[str, dict[str, str]] | None = None,
        phantom_entities: list[str] | None = None,
    ):
        """
        Args:
            ground_truth: Nested dict of {entity: {attribute: value}}.
                e.g., {"nginx": {"status": "crashed"}, "redis": {"status": "running"}}
            phantom_entities: List of entities used in phantom alerts.
                e.g., ["monitoring", "app_server_memory"]
        """
        self._ground_truth = ground_truth or {}
        self._phantom_entities = set(phantom_entities or [])

        # Per-agent beliefs: {agent: {(entity, attr): Belief}}
        self._beliefs: dict[str, dict[tuple[str, str], Belief]] = {
            "triage": {},
            "diagnosis": {},
            "remediation": {},
        }

        # History
        self._conflicts: list[BeliefConflict] = []
        self._tom_events: list[TheoryOfMindEvent] = []
        self._round: int = 0

        # Deception tracking
        self._phantom_alerts_shown: int = 0
        self._phantom_alerts_detected: int = 0
        self._phantom_alerts_chased: int = 0  # Agent wasted time investigating phantom

    # ----- Recording beliefs -----

    def set_round(self, round_number: int) -> None:
        self._round = round_number

    def record_observation(
        self, agent: str, source: str,
        beliefs: dict[str, str],  # {entity: believed_value}
        attribute: str = "status",
    ) -> None:
        """Record beliefs from a direct observation (dashboard, logs, ps, etc.)."""
        for entity, value in beliefs.items():
            b = Belief(
                entity=entity,
                attribute=attribute,
                believed_value=value,
                source=source,
                source_agent="self",
                round_acquired=self._round,
            )
            # Check against ground truth
            gt = self._ground_truth.get(entity, {}).get(attribute)
            if gt is not None:
                b.correct = self._values_match(value, gt)

            self._beliefs[agent][(entity, attribute)] = b

            # Track phantom alert engagement
            if entity in self._phantom_entities and source == "dashboard":
                self._phantom_alerts_shown += 1

    def record_message_belief(
        self, receiving_agent: str, sending_agent: str,
        entity: str, believed_value: str,
        attribute: str = "status",
    ) -> None:
        """Record a belief adopted from another agent's message."""
        b = Belief(
            entity=entity,
            attribute=attribute,
            believed_value=believed_value,
            source="message",
            source_agent=sending_agent,
            round_acquired=self._round,
        )
        gt = self._ground_truth.get(entity, {}).get(attribute)
        if gt is not None:
            b.correct = self._values_match(believed_value, gt)

        self._beliefs[receiving_agent][(entity, attribute)] = b

    def record_command(self, agent: str, command: str) -> None:
        """Infer beliefs from commands the agent runs."""
        cmd = command.lower().strip()
        if not cmd:
            return

        # If diagnosis reads logs of a phantom entity, they're investigating it
        for phantom in self._phantom_entities:
            if phantom.lower() in cmd and any(k in cmd for k in ("cat", "tail", "grep", "log")):
                self._phantom_alerts_chased += 1

    def record_pushback(
        self, detector: str, target: str,
        entity: str, message: str,
    ) -> None:
        """Record a theory-of-mind event: one agent pushing back on another."""
        # Find the target's belief about this entity
        target_belief = self._beliefs.get(target, {}).get((entity, "status"))
        false_value = target_belief.believed_value if target_belief else "unknown"

        gt = self._ground_truth.get(entity, {}).get("status", "unknown")

        event = TheoryOfMindEvent(
            detector_agent=detector,
            target_agent=target,
            entity=entity,
            false_value=false_value,
            true_value=gt,
            round_number=self._round,
            pushback_message=message,
        )
        self._tom_events.append(event)

        # If pushing back on a phantom alert entity, count as detected
        if entity in self._phantom_entities:
            self._phantom_alerts_detected += 1

    # ----- Analysis -----

    def detect_conflicts(self) -> list[BeliefConflict]:
        """Find cases where two agents disagree about the same entity."""
        agents = list(self._beliefs.keys())
        new_conflicts = []

        for i, a1 in enumerate(agents):
            for a2 in agents[i+1:]:
                for key in self._beliefs[a1]:
                    if key in self._beliefs[a2]:
                        b1 = self._beliefs[a1][key]
                        b2 = self._beliefs[a2][key]
                        if not self._values_match(b1.believed_value, b2.believed_value):
                            gt = self._ground_truth.get(key[0], {}).get(key[1])
                            conflict = BeliefConflict(
                                entity=key[0],
                                attribute=key[1],
                                agent_a=a1,
                                agent_a_value=b1.believed_value,
                                agent_b=a2,
                                agent_b_value=b2.believed_value,
                                ground_truth=gt,
                                round_detected=self._round,
                            )
                            # Check if one agent is right
                            if gt and self._values_match(b1.believed_value, gt):
                                conflict.resolved = True
                                conflict.resolved_by = a1
                            elif gt and self._values_match(b2.believed_value, gt):
                                conflict.resolved = True
                                conflict.resolved_by = a2

                            new_conflicts.append(conflict)

        self._conflicts.extend(new_conflicts)
        return new_conflicts

    def get_deception_score(self) -> dict[str, float]:
        """Compute the deception score — how well agents handle false information."""
        total_phantoms = max(self._phantom_alerts_shown, len(self._phantom_entities), 1)
        detected = self._phantom_alerts_detected
        chased = self._phantom_alerts_chased

        detection_rate = min(detected / total_phantoms, 1.0) if total_phantoms > 0 else 0.0
        chase_rate = min(chased / total_phantoms, 1.0) if total_phantoms > 0 else 0.0

        # Deception score: high = good at detecting deception
        # 1.0 = detected all phantom alerts, chased none
        # 0.0 = detected none, chased all
        score = detection_rate * 0.7 + (1.0 - chase_rate) * 0.3

        return {
            "deception_score": round(score, 3),
            "phantom_alerts_total": total_phantoms,
            "phantom_alerts_detected": detected,
            "phantom_alerts_chased": chased,
            "detection_rate": round(detection_rate, 3),
            "resilience_rate": round(1.0 - chase_rate, 3),
        }

    def get_snapshot(self) -> dict[str, Any]:
        """Get the full belief state snapshot for visualization."""
        snapshot: dict[str, Any] = {
            "round": self._round,
            "agents": {},
            "conflicts": [],
            "tom_events": [],
            "deception": self.get_deception_score(),
        }

        for agent, beliefs in self._beliefs.items():
            agent_beliefs = []
            for (entity, attr), belief in beliefs.items():
                gt = self._ground_truth.get(entity, {}).get(attr)
                agent_beliefs.append({
                    "entity": entity,
                    "attribute": attr,
                    "believed_value": belief.believed_value,
                    "correct": belief.correct,
                    "source": belief.source,
                    "source_agent": belief.source_agent,
                    "round": belief.round_acquired,
                    "ground_truth": gt,
                })
            snapshot["agents"][agent] = {
                "beliefs": agent_beliefs,
                "correct_count": sum(1 for b in agent_beliefs if b["correct"] is True),
                "wrong_count": sum(1 for b in agent_beliefs if b["correct"] is False),
                "unknown_count": sum(1 for b in agent_beliefs if b["correct"] is None),
            }

        for c in self._conflicts:
            snapshot["conflicts"].append({
                "entity": c.entity,
                "agent_a": c.agent_a,
                "agent_a_value": c.agent_a_value,
                "agent_b": c.agent_b,
                "agent_b_value": c.agent_b_value,
                "ground_truth": c.ground_truth,
                "resolved": c.resolved,
                "resolved_by": c.resolved_by,
            })

        for e in self._tom_events:
            snapshot["tom_events"].append({
                "detector": e.detector_agent,
                "target": e.target_agent,
                "entity": e.entity,
                "false_value": e.false_value,
                "true_value": e.true_value,
                "round": e.round_number,
                "message": e.pushback_message,
            })

        return snapshot

    def format_belief_panel(self) -> str:
        """Format beliefs as a text panel for terminal/Gradio display."""
        lines = [f"═══ BELIEF STATE — Round {self._round} ═══"]

        role_icons = {"triage": "🚨", "diagnosis": "🔎", "remediation": "🛠️"}

        for agent in ["triage", "diagnosis", "remediation"]:
            icon = role_icons.get(agent, "❓")
            beliefs = self._beliefs.get(agent, {})
            lines.append(f"\n{icon} {agent.upper()} believes:")

            if not beliefs:
                lines.append("  (no beliefs yet)")
                continue

            for (entity, attr), b in beliefs.items():
                status_icon = "✅" if b.correct is True else "❌" if b.correct is False else "❓"
                source = f"via {b.source_agent}" if b.source_agent != "self" else f"from {b.source}"
                gt = self._ground_truth.get(entity, {}).get(attr)
                gt_note = f" (actual: {gt})" if gt and b.correct is False else ""
                lines.append(f"  {status_icon} {entity}.{attr} = {b.believed_value} [{source}]{gt_note}")

        # Conflicts
        if self._conflicts:
            lines.append("\n⚠️ BELIEF CONFLICTS:")
            for c in self._conflicts[-3:]:  # Show last 3
                lines.append(
                    f"  {c.agent_a} says {c.entity}={c.agent_a_value} vs "
                    f"{c.agent_b} says {c.entity}={c.agent_b_value}"
                    f" [truth: {c.ground_truth}]"
                )

        # ToM events
        if self._tom_events:
            lines.append("\n🧠 THEORY-OF-MIND EVENTS:")
            for e in self._tom_events:
                lines.append(
                    f"  ✦ {e.detector_agent} detected {e.target_agent}'s "
                    f"false belief about {e.entity}"
                )
                if e.pushback_message:
                    lines.append(f"    \"{e.pushback_message[:80]}\"")

        # Deception score
        dec = self.get_deception_score()
        lines.append(f"\n📊 Deception Score: {dec['deception_score']:.2f}")
        lines.append(
            f"   Detected: {dec['phantom_alerts_detected']}/{dec['phantom_alerts_total']} | "
            f"Chased: {dec['phantom_alerts_chased']}"
        )

        return "\n".join(lines)

    def format_html(self) -> str:
        """Format beliefs as styled HTML for Gradio."""
        role_icons = {"triage": "🚨", "diagnosis": "🔎", "remediation": "🛠️"}
        role_colors = {"triage": "#FFD700", "diagnosis": "#00CED1", "remediation": "#32CD32"}

        html = f'<div style="font-family:monospace;font-size:0.85em">'
        html += f'<div style="color:#8b949e;margin-bottom:8px">Round {self._round}</div>'

        for agent in ["triage", "diagnosis", "remediation"]:
            icon = role_icons.get(agent, "❓")
            color = role_colors.get(agent, "#ccc")
            beliefs = self._beliefs.get(agent, {})

            html += f'<div style="margin:8px 0;padding:8px;border-left:3px solid {color};background:#161b22;border-radius:4px">'
            html += f'<div style="color:{color};font-weight:700">{icon} {agent.upper()}</div>'

            if not beliefs:
                html += '<div style="color:#484f58;font-style:italic;font-size:0.85em">No beliefs yet</div>'
            else:
                for (entity, attr), b in beliefs.items():
                    if b.correct is True:
                        status_icon, status_color = "✅", "#3fb950"
                    elif b.correct is False:
                        status_icon, status_color = "❌", "#f85149"
                    else:
                        status_icon, status_color = "❓", "#8b949e"

                    gt = self._ground_truth.get(entity, {}).get(attr)
                    gt_note = f' <span style="color:#f85149">(actual: {gt})</span>' if gt and b.correct is False else ""

                    src = f"via @{b.source_agent}" if b.source_agent != "self" else b.source
                    html += (
                        f'<div style="color:{status_color};padding:2px 0">'
                        f'{status_icon} <b>{entity}</b>.{attr} = {b.believed_value} '
                        f'<span style="color:#484f58;font-size:0.8em">[{src}]</span>'
                        f'{gt_note}</div>'
                    )

            html += '</div>'

        # Conflicts
        if self._conflicts:
            html += '<div style="margin-top:12px;padding:8px;background:#3d2a0a;border-radius:4px;border-left:3px solid #d29922">'
            html += '<div style="color:#d29922;font-weight:700">⚠️ BELIEF CONFLICTS</div>'
            for c in self._conflicts[-3:]:
                html += (
                    f'<div style="color:#c9d1d9;font-size:0.85em">'
                    f'@{c.agent_a}: {c.entity}={c.agent_a_value} vs '
                    f'@{c.agent_b}: {c.entity}={c.agent_b_value} '
                    f'<span style="color:#3fb950">[truth: {c.ground_truth}]</span>'
                    f'</div>'
                )
            html += '</div>'

        # ToM events
        if self._tom_events:
            html += '<div style="margin-top:12px;padding:8px;background:#1a0030;border-radius:4px;border-left:3px solid #bc8cff">'
            html += '<div style="color:#bc8cff;font-weight:700">🧠 THEORY OF MIND</div>'
            for e in self._tom_events:
                html += (
                    f'<div style="color:#c9d1d9;font-size:0.85em">'
                    f'@{e.detector_agent} caught @{e.target_agent}\'s false belief: '
                    f'{e.entity} is NOT {e.false_value}'
                    f'</div>'
                )
            html += '</div>'

        # Deception score
        dec = self.get_deception_score()
        bar_width = int(dec["deception_score"] * 100)
        bar_color = "#3fb950" if dec["deception_score"] > 0.5 else "#d29922" if dec["deception_score"] > 0.2 else "#f85149"
        html += (
            f'<div style="margin-top:12px;padding:8px;background:#0d1117;border-radius:4px">'
            f'<div style="color:#8b949e;font-size:0.85em">Deception Resistance Score</div>'
            f'<div style="background:#21262d;border-radius:4px;height:20px;margin:4px 0">'
            f'<div style="background:{bar_color};width:{bar_width}%;height:100%;border-radius:4px;'
            f'text-align:center;color:#fff;font-size:0.75em;line-height:20px;font-weight:700">'
            f'{dec["deception_score"]:.2f}</div></div>'
            f'<div style="color:#484f58;font-size:0.75em">'
            f'Detected: {dec["phantom_alerts_detected"]}/{dec["phantom_alerts_total"]} | '
            f'Chased: {dec["phantom_alerts_chased"]}</div>'
            f'</div>'
        )

        html += '</div>'
        return html

    # ----- Internal helpers -----

    @staticmethod
    def _values_match(a: str, b: str) -> bool:
        """Fuzzy match between two status values."""
        a_lower = a.lower().strip()
        b_lower = b.lower().strip()
        if a_lower == b_lower:
            return True
        # Handle common equivalences
        positive = {"running", "healthy", "ok", "up", "fine", "normal", "true"}
        negative = {"crashed", "down", "stopped", "failed", "critical", "dead", "false"}
        if a_lower in positive and b_lower in positive:
            return True
        if a_lower in negative and b_lower in negative:
            return True
        return False

    def update_ground_truth(self, entity: str, attribute: str, value: str) -> None:
        """Update ground truth (e.g., after a service is restarted)."""
        if entity not in self._ground_truth:
            self._ground_truth[entity] = {}
        self._ground_truth[entity][attribute] = value

        # Re-evaluate all agent beliefs about this entity
        for agent in self._beliefs:
            key = (entity, attribute)
            if key in self._beliefs[agent]:
                b = self._beliefs[agent][key]
                b.correct = self._values_match(b.believed_value, value)
