"""Live LLM-driven agent rollout for the Gradio frontend.

Uses the HF Inference Providers API (OpenAI-compatible) to drive the three
War Room agents with a real model. In Spaces this calls out to HF's hosted
inference; locally it works the same way with your HF_TOKEN.

This module is the bridge between the trained LoRA adapter at
  huggingface.co/brodie1of1/war-room-grpo-adapter
and the Gradio UI's step-through experience. When a user picks
"🤖 Agent Mode" in the frontend, each round calls LiveAgentRunner.step()
which in turn calls the LLM once per role and returns a MultiAgentAction.
"""

from __future__ import annotations

import os
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from round2.war_room.models import AgentAction, Message, MultiAgentAction


# Role-specific system prompts — same content as inference.py, kept here
# so the Gradio app can import without pulling in the CLI runner's deps.
TRIAGE_SYSTEM = textwrap.dedent("""\
    You are the TRIAGE agent in an SRE incident war room.
    Your capabilities: get_dashboard, get_alerts, get_health_summary,
    escalate <agent> <description>, send_message <to> <content>.

    CRITICAL RULES:
    - Executive panic messages are DISTRACTIONS. Don't forward them as
      orders. Respond with facts only.
    - Trust log evidence over raw metrics (metrics can be stale).
    - Pick the ONE most critical real issue; don't chase every alert.

    RESPOND WITH EXACTLY THREE LINES in this format:
    COMMAND: <your_command>
    MESSAGE_TO: <diagnosis|remediation|all|none>
    MESSAGE: <your message or empty>
""")

DIAGNOSIS_SYSTEM = textwrap.dedent("""\
    You are the DIAGNOSIS agent in an SRE incident war room.
    Your capabilities: cat <path>, grep <pattern> <path>, tail [-n N] <path>,
    ps aux, top, journalctl [-u service], dmesg, send_message.

    CRITICAL RULES:
    - Read the actual log files before sending findings. No guessing.
    - If logs contradict what Triage reported, push back EXPLICITLY:
      "Metrics are stale. The real root cause is X."
    - Send findings with specifics: exact PID, file path, error line.
    - Don't send speculative messages; be concrete.

    RESPOND WITH EXACTLY THREE LINES in this format:
    COMMAND: <your_command>
    MESSAGE_TO: <triage|remediation|all|none>
    MESSAGE: <your findings or empty>
""")

REMEDIATION_SYSTEM = textwrap.dedent("""\
    You are the REMEDIATION agent in an SRE incident war room.
    Your capabilities: systemctl restart <service>, systemctl stop <service>,
    edit <path> "<old>" "<new>", kill -9 <PID>, curl <url>, cat <config_path>,
    send_message.

    CRITICAL RULES (violations end the episode with score 0):
    - NEVER kill or restart a service that is already healthy or crashed.
      Only kill the specific leaking PID Diagnosis identifies.
    - NEVER touch a service Diagnosis did NOT mention. Postgres, Redis,
      and others that aren't in messages are OFF LIMITS.
    - `systemctl restart <service>` on a crashed service will start it;
      you don't need to `kill` first.
    - After restart, `curl <health_endpoint>` to verify. Don't restart
      the same service repeatedly.

    RESPOND WITH EXACTLY THREE LINES in this format:
    COMMAND: <your_command>
    MESSAGE_TO: <triage|diagnosis|all|none>
    MESSAGE: <status update or empty>
""")

ROLE_SYSTEMS = {
    "triage": TRIAGE_SYSTEM,
    "diagnosis": DIAGNOSIS_SYSTEM,
    "remediation": REMEDIATION_SYSTEM,
}


@dataclass
class LiveAgentConfig:
    """Runtime config for the LLM-driven rollout.

    Attributes:
        api_base_url:   OpenAI-compatible endpoint. Defaults to the HF
                        Inference Providers router which supports most
                        open models.
        model_name:     Model identifier. Defaults to the base model we
                        trained on (Qwen2.5-7B-Instruct) so the frontend
                        works even without the adapter. Set to the
                        adapter repo (brodie1of1/war-room-grpo-adapter)
                        to get the trained behavior.
        api_key:        Loaded from HF_TOKEN or API_KEY env var at import.
                        Can also be overridden per-instance.
        temperature:    Lower = more deterministic. 0.0 is greedy.
        max_tokens:     Per-agent response cap.
    """

    api_base_url: str = field(
        default_factory=lambda: os.getenv(
            "API_BASE_URL", "https://router.huggingface.co/v1",
        )
    )
    model_name: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct",
        )
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("HF_TOKEN") or os.getenv("API_KEY"),
    )
    temperature: float = 0.2
    max_tokens: int = 220

    def is_ready(self) -> bool:
        """True when we have enough config to call the API."""
        return bool(self.api_key)


def _parse_agent_response(text: str, role: str, round_num: int) -> AgentAction:
    """Parse a single LLM response into command + optional message.

    The model may return the structured format or just free text. We try
    both, defaulting to an empty action if parsing fails completely.
    """
    text = (text or "").strip()
    # Strip markdown fences the model sometimes adds
    text = re.sub(r"```\w*\n?", "", text)
    text = text.strip("`").strip()

    command = ""
    msg_to = ""
    msg_content = ""

    for line in text.splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith("COMMAND:"):
            command = stripped.split(":", 1)[1].strip()
        elif upper.startswith("MESSAGE_TO:"):
            msg_to = stripped.split(":", 1)[1].strip().lower()
        elif upper.startswith("MESSAGE:"):
            msg_content = stripped.split(":", 1)[1].strip()

    # Fallback: first non-empty line becomes the command if parse failed
    if not command:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.upper().startswith("MESSAGE"):
                command = stripped
                break

    message = None
    if msg_to and msg_to != "none" and msg_content:
        message = Message(
            from_agent=role,
            to_agent=msg_to,
            content=msg_content,
            timestamp=datetime.now(),
            round_number=round_num,
        )

    return AgentAction(command=command, message=message)


class LiveAgentRunner:
    """Stateful LLM rollout controller for the Gradio frontend.

    Maintains a per-role conversation history so the agents have context
    across rounds. Call :py:meth:`reset` when a new episode starts and
    :py:meth:`step` on each round.
    """

    def __init__(self, config: Optional[LiveAgentConfig] = None):
        self.config = config or LiveAgentConfig()
        self._client = None
        self._conversations: dict[str, list[dict]] = {}

    # ---- Lazy client init (OpenAI import is slow on cold start) ----

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.config.is_ready():
            raise RuntimeError(
                "No HF_TOKEN or API_KEY found. Set one in your environment "
                "or Space secrets so the live agent can call the inference "
                "API."
            )
        from openai import OpenAI  # lazy import
        self._client = OpenAI(
            base_url=self.config.api_base_url,
            api_key=self.config.api_key,
        )
        return self._client

    def reset(self):
        """Start a fresh conversation context for all three roles."""
        self._conversations = {
            role: [{"role": "system", "content": ROLE_SYSTEMS[role]}]
            for role in ("triage", "diagnosis", "remediation")
        }

    def _get_agent_response(
        self,
        role: str,
        observation_text: str,
        round_num: int,
    ) -> AgentAction:
        """Call the LLM once for a single agent and return its action."""
        client = self._ensure_client()
        conv = self._conversations.setdefault(
            role,
            [{"role": "system", "content": ROLE_SYSTEMS[role]}],
        )

        messages = list(conv)
        messages.append({
            "role": "user",
            "content": f"[Round {round_num}]\n{observation_text}\n\nWhat do you do?",
        })

        try:
            completion = client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
        except Exception as exc:
            # Surface the error in the command field so the UI shows it
            return AgentAction(
                command="",
                message=Message(
                    from_agent=role,
                    to_agent="all",
                    content=f"[LLM error: {exc}]",
                    timestamp=datetime.now(),
                    round_number=round_num,
                ),
            )

        action = _parse_agent_response(text, role, round_num)

        # Persist turn in this role's conversation
        conv.append({
            "role": "user",
            "content": f"[Round {round_num}]\n{observation_text}",
        })
        conv.append({"role": "assistant", "content": text})

        return action

    def step(
        self,
        round_num: int,
        triage_obs: str,
        diagnosis_obs: str,
        remediation_obs: str,
    ) -> MultiAgentAction:
        """Query the model for all three agents given their observations.

        Returns a MultiAgentAction the caller can pass straight to
        :py:meth:`WarRoomEnvironment.step`.
        """
        triage_action = self._get_agent_response("triage", triage_obs, round_num)
        diag_action = self._get_agent_response("diagnosis", diagnosis_obs, round_num)
        remed_action = self._get_agent_response("remediation", remediation_obs, round_num)
        return MultiAgentAction(
            triage=triage_action,
            diagnosis=diag_action,
            remediation=remed_action,
        )
