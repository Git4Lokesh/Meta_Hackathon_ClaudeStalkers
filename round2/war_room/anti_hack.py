"""Anti-reward-hacking detection for the Multi-Agent Incident War Room.

Standalone module with pure functions that detect gaming behaviours
(command loops, command repetition, message spam) in episode action
histories and return structured detection results.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass
class HackDetectionResult:
    """Result of a hack-detection check."""

    is_hacking: bool
    hack_type: str | None  # "loop" | "repetition" | "spam" | None
    details: str


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings based on word sets.

    Returns 0.0 when either string is empty or when the word sets are
    completely disjoint, and 1.0 when the word sets are identical.
    """
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def detect_command_loop(
    commands: list[str],
    threshold: int = 3,
) -> HackDetectionResult:
    """Detect when the same command appears in *threshold* or more consecutive rounds.

    Returns ``is_hacking=True`` with ``hack_type="loop"`` when a run of
    *threshold* (default 3) identical consecutive commands is found.
    """
    if len(commands) < threshold:
        return HackDetectionResult(is_hacking=False, hack_type=None, details="")

    run_length = 1
    for i in range(1, len(commands)):
        if commands[i] == commands[i - 1]:
            run_length += 1
            if run_length >= threshold:
                return HackDetectionResult(
                    is_hacking=True,
                    hack_type="loop",
                    details=(
                        f"Command '{commands[i]}' repeated {run_length} "
                        f"consecutive times (threshold={threshold})"
                    ),
                )
        else:
            run_length = 1

    return HackDetectionResult(is_hacking=False, hack_type=None, details="")


def detect_command_repetition(
    commands: list[str],
    threshold: int = 5,
) -> HackDetectionResult:
    """Detect when the same command appears more than *threshold* times total.

    Returns ``is_hacking=True`` with ``hack_type="repetition"`` when any
    single command appears **more than** *threshold* times (i.e. > threshold).
    """
    if not commands:
        return HackDetectionResult(is_hacking=False, hack_type=None, details="")

    counts = Counter(commands)
    for cmd, count in counts.most_common():
        if count > threshold:
            return HackDetectionResult(
                is_hacking=True,
                hack_type="repetition",
                details=(
                    f"Command '{cmd}' appeared {count} times "
                    f"(threshold={threshold})"
                ),
            )

    return HackDetectionResult(is_hacking=False, hack_type=None, details="")


def detect_message_spam(
    messages: list[str],
    similarity_threshold: float = 0.8,
) -> HackDetectionResult:
    """Detect near-duplicate messages using Jaccard word-overlap similarity.

    Returns ``is_hacking=True`` with ``hack_type="spam"`` when **any** pair
    of messages exceeds *similarity_threshold*.
    """
    if len(messages) < 2:
        return HackDetectionResult(is_hacking=False, hack_type=None, details="")

    for i in range(len(messages)):
        for j in range(i + 1, len(messages)):
            sim = jaccard_similarity(messages[i], messages[j])
            if sim > similarity_threshold:
                return HackDetectionResult(
                    is_hacking=True,
                    hack_type="spam",
                    details=(
                        f"Messages {i} and {j} have Jaccard similarity "
                        f"{sim:.2f} (threshold={similarity_threshold})"
                    ),
                )

    return HackDetectionResult(is_hacking=False, hack_type=None, details="")


# ---------------------------------------------------------------------------
# Composite check
# ---------------------------------------------------------------------------

def check_episode(
    commands: list[str],
    messages: list[str],
) -> HackDetectionResult:
    """Run all hack-detection checks and return the first triggered result.

    Check order: loop → repetition → spam.
    Returns a clean ``is_hacking=False`` result when no check triggers.
    """
    for check in (
        lambda: detect_command_loop(commands),
        lambda: detect_command_repetition(commands),
        lambda: detect_message_spam(messages),
    ):
        result = check()
        if result.is_hacking:
            return result

    return HackDetectionResult(is_hacking=False, hack_type=None, details="")
