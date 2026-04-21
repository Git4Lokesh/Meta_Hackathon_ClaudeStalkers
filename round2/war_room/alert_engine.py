"""AlertEngine for the Multi-Agent Incident War Room."""

from __future__ import annotations

from round2.war_room.models import Alert
from sre_env.server.simulated_system import SimulatedSystem


class AlertEngine:
    def __init__(
        self,
        prominence_overrides: dict[str, int] | None = None,
        phantom_alerts: list[Alert] | None = None,
    ):
        """
        prominence_overrides: {service_name: prominence_value} for misdirection.
        phantom_alerts: Fake alerts injected into the dashboard that don't reflect
                       actual system state. Tests theory-of-mind — diagnosis must
                       recognize these are false and push back on triage.
        """
        self._alerts: list[Alert] = []
        self._prominence_overrides = prominence_overrides or {}
        self._phantom_alerts = phantom_alerts or []
        self._real_alerts: list[Alert] = []  # Track which are real vs phantom

    def evaluate(self, system: SimulatedSystem) -> list[Alert]:
        """Generate alerts from current system state."""
        alerts = []

        for name, svc in system.service_registry.services.items():
            prominence = self._prominence_overrides.get(name, 0)

            if svc.status == "crashed":
                alerts.append(Alert(
                    service=name, alert_type="service_down",
                    severity="critical",
                    description=f"Service {name} is DOWN (crashed)",
                    prominence=prominence,
                ))
            elif svc.status == "degraded":
                alerts.append(Alert(
                    service=name, alert_type="service_degraded",
                    severity="warning",
                    description=f"Service {name} is degraded",
                    prominence=prominence,
                ))

            # Check metrics
            latest = system.metrics_store.latest(name)
            if latest:
                if latest.cpu_percent > 80:
                    alerts.append(Alert(
                        service=name, alert_type="high_cpu",
                        severity="warning",
                        description=f"High CPU on {name}: {latest.cpu_percent:.1f}%",
                        prominence=prominence,
                    ))
                if latest.memory_percent > 70:
                    alerts.append(Alert(
                        service=name, alert_type="high_memory",
                        severity="warning",
                        description=f"High memory on {name}: {latest.memory_percent:.1f}%",
                        prominence=prominence,
                    ))

        self._real_alerts = list(alerts)  # Store real alerts separately

        # Inject phantom alerts (stale/cached metrics that aren't real)
        for phantom in self._phantom_alerts:
            alerts.append(phantom)

        # Sort by prominence (desc) then severity (critical > warning > info)
        severity_order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda a: (-a.prominence, severity_order.get(a.severity, 3)))

        self._alerts = alerts
        return alerts

    def get_active_alerts(self) -> list[Alert]:
        return list(self._alerts)

    def is_phantom_alert(self, alert: Alert) -> bool:
        """Check if an alert is a phantom (fake/stale)."""
        return alert not in self._real_alerts

    def get_phantom_count(self) -> int:
        return len(self._phantom_alerts)

    def get_dashboard_summary(self) -> str:
        """Format alerts as a dashboard text summary."""
        if not self._alerts:
            return "Dashboard: All systems operational. No active alerts."

        lines = ["=== MONITORING DASHBOARD ==="]

        # Group by severity
        for severity in ["critical", "warning", "info"]:
            sev_alerts = [a for a in self._alerts if a.severity == severity]
            if sev_alerts:
                icon = "🔴" if severity == "critical" else "🟡" if severity == "warning" else "🔵"
                lines.append(f"\n{icon} {severity.upper()} ({len(sev_alerts)}):")
                for a in sev_alerts:
                    lines.append(f"  • [{a.service}] {a.description}")

        return "\n".join(lines)

    def format_alerts(self) -> str:
        """Format alerts as a detailed list."""
        if not self._alerts:
            return "No active alerts."
        lines = []
        for i, a in enumerate(self._alerts, 1):
            lines.append(f"{i}. [{a.severity.upper()}] {a.service}: {a.description} (type: {a.alert_type})")
        return "\n".join(lines)
