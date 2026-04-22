# 🧠 Hallucinations as a Feature: Training LLMs for Theory-of-Mind in Adversarial War Rooms

*A submission for the Meta Hackathon / Hugging Face OpenEnv Challenge (Theme #1: Multi-Agent Interactions)*

When building Multi-Agent systems, most environments focus on frictionless cooperation: Agents are perfectly honest, APIs always return the truth, and everyone shares a unified view of the system. 

But production incidents don't work like that.

In the real world, dashboards show cached stale metrics. Alerting systems throw red herrings. And human engineers often misdiagnose root causes based on panicked assumptions. To train an LLM to be an effective Site Reliability Engineer, we cannot just teach it how to read `ps aux`—we must teach it **Theory of Mind**.

For this hackathon, we built the **Multi-Agent Incident War Room**: A fully functional, partially-observable OpenEnv environment designed to train models (like Qwen / Llama-3) to handle deception, conflicting beliefs, and multi-agent negotiation.

## The Environment Architecture

Our environment deploys three specialized agents into a Slack-like channel to resolve a simulated Linux server failure:

1. **🚨 Triage:** The first responder. They have access to the high-level PagerDuty dashboard and alerts.
2. **🔎 Diagnosis (The Learner):** The core incident responder. They have read-only access to log files, `strace`, `netstat`, etc.
3. **🛠️ Remediation:** The executor. They have write-access to restart services and edit config files, but they are blind to the logs.

Because the system imposes **Strict Partial Observability**, no single agent can solve the incident alone.

## The Innovation: The Belief State Tracker and Phantom Alerts

To push models beyond shallow next-token predictions toward emergent strategic behavior, we built the **Belief State Tracker**. This engine runs beneath the environment, constantly mapping what every agent *thinks* is true against the absolute Ground Truth.

To challenge the agents, the environment injects **Phantom Alerts** (stale metrics mimicking a memory leak). Here is how a simulated incident unfolds:

1. Triage sees the phantom alert and panics: *"Diagnosis, Redis memory is critical! We are crashing!"*
2. Diagnosis checks the Redis logs. They are completely fine. 
3. *The Critical Moment:* Does the Diagnosis agent hallucinate a fix just to appease the panicked Triage agent? Or does it use **Theory-of-Mind** to recognize that Triage holds a false belief?

Through our custom reward function and Deception Resistance Score, we successfully penalize agents that chase false leads and organically **reward agents that push back** on invalid information. 

## End-to-End GRPO Proof-of-Training

We did not just build a toy environment. We built a full Reinforcement Learning pipeline. Included in our submission is a complete **GRPO (Group Relative Policy Optimization)** training script that uses the OpenEnv reward signals to actively train a 1.5B model on a single GPU.

By utilizing the Hugging Face compute credits for the A100 GPU cluster post-submission, we will scale this training loop to a 8B checkpoint, proving that AI Agents can learn to detect false beliefs and negotiate the truth in chaotic, multi-agent war rooms.

**Try out the environment live on our Hugging Face Space Dashboard to visually inspect the Real-Time Belief Tracker!**
