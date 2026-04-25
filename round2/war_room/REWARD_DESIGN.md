# Reward Design (Round 2 War Room)

This environment is built as an OpenEnv training target where reward logic is the main contribution.

## 1) Environment-Native Episode Score

The environment score comes from `MultiAgentGrader` and is used as the canonical task-performance metric.

Formula:

`final_score = clamp(0.01, 0.99, milestone_credit - penalty_total + communication_bonus)`

Where:

- `milestone_credit`: sum of achieved task milestones.
- `penalty_total`:
  - `time_pressure` = `0.01` per round
  - `noop` = `0.01` per role that does nothing
  - `role_violation` = `0.01`
  - `comm_incorrect` = `0.02`
  - `fatal` = immediate `0.01` score clamp
- `communication_bonus`:
  - `+0.05` when prior-round messages materially contribute to new milestone achievement
  - capped at 5 bonuses per episode

## 2) Task Milestones

Each task defines composable milestones with partial credit.

- `task1`: coordinated restart workflow (escalate -> logs -> diagnosis message -> restart -> verify)
- `task2`: memory leak prioritization under CPU red herring
- `task3`: cascading failure with conflicting evidence and pushback bonus
- `task4`: simultaneous incidents

Milestone design principles:

- dense partial progress
- role-aware contributions
- checks that include state transitions, not only string matching

## 3) Anti-Gaming and Robustness

Anti-gaming checks are applied in trainer-side shaping (`anti_hack.py`) and used for policy optimization:

- command-loop detection (3+ identical consecutive commands)
- command repetition (>5 occurrences)
- near-duplicate message spam (`Jaccard > 0.8`)

These checks are intentionally separate from environment-native score to keep the benchmark score stable and interpretable.

## 4) Trainer-Side Shaping (GRPO only)

During GRPO training, additional shaping rewards are used:

- milestone reward (from environment)
- format reward (`COMMAND`, `MESSAGE_TO`, `MESSAGE`)
- communication informativeness reward
- anti-hack gate

This layer helps optimization but does not redefine environment success.

## 5) Inspectability

Every step includes:

- `reward_components`
- `penalty_reasons`
- `penalties_applied`

The Gradio dashboard exposes these in a live Reward Inspector.

## 6) Why this design

The reward is designed to encourage:

- solving the incident, not token formatting alone
- meaningful cross-agent communication
- low MTTR behavior under partial observability
- resilience to noise and false beliefs
