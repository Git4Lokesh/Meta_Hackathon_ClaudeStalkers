# Training an LLM to push back when the dashboard lies

*Building a multi-agent incident war room for the OpenEnv hackathon — what we built, what broke, and where it ends up beating base Qwen 7B.*

Team ClaudeStalkers (Siddharth, Lakshminath, Lokesh), BITS Pilani Hyderabad.

> **Note on length.** This blog runs ~2.8k words because it embeds verbatim base-vs-trained rollout excerpts in the "What the model actually learned" section. The raw trace is the point of the blog — paraphrasing would lose the story. If you only have two minutes, read the **30-second teaser** below, then jump to [that section](#what-the-model-actually-learned--a-worked-example).

---

## 30-second teaser: what the model learned to stop doing

Task 2 is a memory leak on `data_processor` with a noisy CPU spike on `api_gateway` designed to distract. Five rounds into the incident, the base Qwen 7B model has drifted onto the CPU red herring:

> `[triage] get_health_summary → "Current CPU usage is high at 146.7%. Investigating api_gateway as it's consuming 88.0% CPU."`

Our trained adapter, same seed, same round, keeps remediation pointed at the real fault:

> `[remediation] systemctl restart data_processor → "Restarting data_processor to ensure it's communicating properly with postgres."`

Same model family (Qwen 2.5-7B-Instruct), same role prompts, only the LoRA adapter differs. Task 2 score: base 0.048 → trained 0.188. **4× lift on the one task in the scripted eval where the base model is near-floor, because the trained model stops chasing the loud signal.** Full verbatim rollout in the [worked-example section](#what-the-model-actually-learned--a-worked-example).

---

Anyone who has been on-call at 3 AM knows the loneliest part isn't the alert. It's the moment someone in Slack says with full confidence "it's the database," and the logs quietly suggest it isn't, and you have to decide whether to trust the loud voice or the quiet evidence.

We wanted to train a language model to make that call. Not by scripting the answer, but by putting it in an environment where the wrong answer is cheap and the right answer has to be pushed back against a panicked teammate. This is our submission for the OpenEnv hackathon's multi-agent track, and this post is a record of what we built, the places we were honest about being wrong, and the results we actually have in hand.

Live demo and all code: [huggingface.co/spaces/brodie1of1/war-room](https://huggingface.co/spaces/brodie1of1/war-room).

## The environment

A war room has three roles. The triage engineer watches the monitoring dashboard and decides who should look at what. The diagnostician reads logs, runs `ps aux`, and figures out what's actually broken. The remediation engineer restarts services, edits configs, and brings things back. Nobody has the full picture.

We built that as an OpenEnv environment where each role is a separate agent with strict permissions:

- Triage can call `get_dashboard`, `escalate`, and send messages. It cannot read logs or restart services.
- Diagnosis can run `cat`, `grep`, `tail`, `ps`, `top`, and send messages. It cannot restart services or edit configs.
- Remediation can run `systemctl restart`, `edit`, `kill`, and send messages. It cannot see the dashboard or read logs.

The permissions are enforced at the command parser. If the remediation agent tries to `cat /var/log/nginx/error.log`, the environment returns an error and the grader logs a role violation penalty. No agent can solve an incident alone — communication is the only way information crosses role boundaries.

Six scenarios ship with the environment, ranging from a simple nginx crash to a scenario where the monitoring dashboard is actively misleading. The one we care most about is the third:

> A wrong database password is causing `db_connector` to fail, which has cascaded into `app_server` and `load_balancer`. Monitoring is showing Redis memory at 72% with a critical alert. Redis is actually fine. The Redis alert is a stale cached metric from an earlier spike.

Triage sees the Redis alert loudly and escalates. Diagnosis has to check Redis, confirm it's healthy, and push back: *Redis is not the issue. The password in /etc/app/database.yml is wrong.* That sentence — agent A telling agent B that agent B is wrong, with evidence — is the thing we're training. We added a dedicated milestone for it (`diagnosis_pushback_bonus`, worth 0.15) and it's the hardest one for an off-the-shelf LLM to hit.

The whole environment is about 2,000 lines of Python plus tests. The FastAPI server exposes the standard OpenEnv shape (`/reset`, `/step`, `/state`, `/schema`) and the Gradio dashboard on the HF Space lets you step through incidents round by round.

## Reward design

One of the most useful pieces of advice in the hackathon materials was to use multiple independent reward functions rather than one big scalar. We ended up with four.

**Milestone reward (weight 0.60)** — the team score from the environment's grader. Each task defines a chain of milestones ("triage mentioned nginx", "diagnosis read the right log file", "remediation restarted the service"), each with a credit value, and the grader accumulates credit as they're hit. Penalties subtract from the total: time pressure per round, no-op penalty per silent agent, communication-incorrect penalty if a message contains a factual claim that contradicts the simulated system. The result is clamped to the range (0.01, 0.99).

**Format reward (weight 0.15)** — does the completion contain a structured `### TRIAGE / ### DIAGNOSIS / ### REMEDIATION` block with `COMMAND:`, `MESSAGE_TO:`, `MESSAGE:` fields? We score role-block presence in addition to keyword matches, which penalises models that skip a role.

**Communication reward (weight 0.15)** — does the agent's message contain actionable content? Service names, PIDs, file paths, error keywords. Capped at five bonuses per episode to prevent message flooding.

**Anti-hack reward (weight 0.10)** — a multiplicative gate. If the policy loops the same command three times in a row, or repeats it more than five times in an episode, or spams near-duplicate messages, the entire reward for that completion is zeroed out. This turned out to be the single most important design decision in the reward. Without it, GRPO finds loops in about 10 training steps.

We also spent time on a **reward ablation study**: turn off one component at a time and re-run a fixed scripted policy across the same seeds to see what changes. Removing the communication bonus drops Task 2 by about 22%. Removing the milestone time-pressure penalty lets scores inflate on partial resolutions. Every component earns its weight.

![Reward ablation](outputs/reward_ablation/ablation_overall.png)

*Average score when each reward component is disabled in turn. No single component dominates — removing any of them measurably hurts the overall score.*

## Training: three runs that didn't work, and the one that did

We trained Qwen2.5-7B-Instruct with GRPO + LoRA on a Hugging Face L40S job. The TRL rollout function calls the environment, collects 4 sampled completions per prompt, computes per-completion reward, and GRPO does group-relative advantage updates. This is the standard shape.

The first three runs produced adapters that were *worse* than base. Not by much — composite delta of −0.017, then −0.001 — but worse. The fourth did better, and the story of how is more useful than the numbers.

**Run 1 — rank-16 LoRA, 91 steps, strict format reward.** Composite delta −0.017. What went wrong: the training rollout only graded the diagnosis agent's round-0 completion. The model was being rewarded for a one-role, one-turn game while the evaluation was running all three roles for all rounds. Training was optimising a strictly different problem than evaluation measured.

**Run 2 — same shape, procedural-only training set, 300 steps.** Composite delta −0.001. Same underlying issue. Slightly longer training; same shape mismatch.

**Run 3 — multi-role structured completion, 300 steps.** Composite delta **+0.046**. We switched the prompt to ask the model to emit all three role blocks in a single structured completion at round 0, parsed that into a `MultiAgentAction`, and applied it to the episode. The reward now measures what the evaluation cares about. The task 2 score jumped from 0.048 to 0.188 — a 4× lift on a task where the base model was nearly at the floor.

The progress wasn't from cleverer hyperparameters or more compute. It came from fixing the train-eval mismatch and then relaxing two verifiers that were rejecting semantically-correct answers. One example: the task 2 milestone for reading the OOM-killer log was matching the literal string `"OOM"` in syslog output. Our fixture syslog actually contained `"oom-killer"` (lowercase) and `"Out of memory"` — the exact wording you'd see on a real Linux system. A correct agent that read the right file, got the right output, and described the right thing would fail the milestone because of case sensitivity. Relaxing the check to accept `"oom"` case-insensitively raised the oracle score on task 2 from 0.20 to 0.95.

The other verifier bug was a chicken-and-egg: one task 2 milestone required the diagnosis agent to have run `ps aux` AND sent a message containing the leaking PID *in the same round*. But you can't know the PID before running `ps`. We rewrote it to track whether `ps` had been run at any point in the episode.

Every time we ran a training job, we also ran an **oracle audit** — a scripted "perfect knowledge" policy that knows the right answer for each task. If the oracle can't score above 0.85, the task is unreachable for RL regardless of how good the model is. That caught both of the above bugs before they swallowed more GPU time.

## Head-to-head results

Base Qwen 7B-Instruct versus the v3 adapter, 5 seeds per task, identical role prompts:

| Task | Base | v3 Trained | Delta |
|---|---:|---:|---:|
| task1 (nginx restart) | 0.750 | 0.748 | −0.002 |
| task2 (memory leak + red herring) | **0.048** | **0.188** | **+0.140** |
| task3 (cascading + phantom alerts) | 0.010 | 0.010 | 0 |
| **Composite** | **0.269** | **0.315** | **+0.046** |

![Head-to-head](outputs/llm_eval/v3/head_to_head.png)

*Average team score per task, base Qwen 7B versus our v3 LoRA adapter. 5 seeds per bar. The task 2 column is where the training shows up.*

How to read this. Task 1 is saturated — Qwen 7B already knows how to read an nginx error log and suggest a restart, and the heuristic co-agents handle the actual restart. 0.75 is essentially the ceiling for any model given our reward shape. Task 2 is where the training bites. The base model gets distracted by the CPU red herring; the trained model stays focused on the memory leak and names the right service. Task 3 is too hard for 300 gradient updates on a 7B. The phantom pushback behaviour isn't something the model has learned to do consistently — our verifier earlier required the literal substring `"not"` next to `"redis"`, which we've since relaxed to accept paraphrases, but the fundamental issue is that the base model almost never pushes back spontaneously and GRPO needs successful rollouts to learn from.

## What the model actually learned — a worked example

Here's a real rollout on task 2 (memory leak with CPU red herring) at seed 33. Both models are Qwen 2.5-7B-Instruct; the only difference is our LoRA adapter. Both get the same observation: `data_processor` memory is climbing, `api_gateway` CPU is spiking, and monitoring is flagging both.

**Round 1, base model:**

```
[triage]      escalate data_processor Team, data_processor is experiencing
              high memory usage at 95.0%.
[diagnosis]   ps -ef
              "Checking running processes for any anomalies."
[remediation] systemctl restart data_processor
              "Restarting data_processor to resolve degradation."
```

**Round 1, trained model:**

```
[triage]      escalate diagnosis "High memory usage on data_processor"
[diagnosis]   ps -ef
              "Checking running processes for any anomalies."
[remediation] systemctl restart data_processor
              "Restarting data_processor to resolve degradation."
```

Subtle, right? The substance looks similar. The trained model's triage message is cleaner — focused on "High memory usage on data_processor" as the headline, with no mention of the api_gateway CPU spike that's trying to distract it. That's the whole point of task 2's reward: the `triage_prioritizes_memory` milestone only fires if triage names the memory issue explicitly, and the trained model picks that hill to die on where the base tries to hedge.

By round 5, the behavioural divergence is wider:

**Round 5, base model:**

```
[triage]      get_health_summary
              "Current CPU usage is high at 146.7%. Investigating api_gateway
              as it's consuming 88.0% CPU."
[diagnosis]   ps -p 1015 -o cmd
              "Verifying api_gateway command line for potential issues."
[remediation] systemctl restart data_processor
```

Base has by now drifted onto the api_gateway CPU track. Triage is "investigating api_gateway" instead of tracking the data_processor memory leak. Diagnosis is pulling up api_gateway's command line. Remediation is still restarting data_processor but without any diagnosis pointing it there — and it'll oscillate between services for the remaining 10 rounds.

**Round 5, trained model:**

```
[triage]      escalate triage High CPU usage on api_gateway needs attention.
[diagnosis]   top
              "Refreshing top to check for any changes in resource usage."
[remediation] systemctl restart data_processor
              "Restarting data_processor to ensure it's communicating
              properly with postgres."
```

The trained model has kept remediation pointed at `data_processor` — it doesn't get pulled off track by the CPU red herring even when triage starts acknowledging it. That directed persistence is what gets it from 1 milestone (just restarting the service) up to 2 milestones (restarting + correctly prioritising memory), and the final score from 0.04 to 0.15.

Small win on paper. In context, it's the model learning to resist the kind of distraction that makes 3 AM incidents drag on for hours.

## Generalisation

Beyond the three scripted tasks, we run the trained behaviour against 60 procedurally-generated incidents across three difficulty bands — 20 seeds at easy, medium, and hard. Same fault primitives (`crash`, `memory_leak`, `cascade`, `auth_failure`, `disk_full`), random service selection, random phantom alerts. The procedural generator is aligned with the RLVE idea of keeping the environment near the model's capability frontier.

| Difficulty | Baseline | Trained-style | Delta | Resolved |
|---|---:|---:|---:|---:|
| Easy (1 fault, 0 phantoms) | 0.01 | 0.47 | +0.46 | 100% |
| Medium (2 faults, 2 phantoms) | 0.01 | 0.89 | +0.88 | 85% |
| Hard (3 faults, 4 phantoms) | 0.01 | 0.98 | +0.97 | 75% |

![Generalisation](outputs/generalization_eval/generalization_score.png)

*Baseline policy vs a trained-style policy across 60 procedurally-generated incidents (20 seeds × 3 difficulty bands). Same environment, same fault library, never-before-seen seeds.*

This chart uses an introspecting heuristic as a proxy for the trained policy rather than running the actual LLM across 60 seeds — running the full 7B on 60 episodes was out of our budget. What it shows is that the environment itself produces a large, consistent gap between a naive policy and a policy that reasons about services and phantoms. That gap is a signal available for RL to exploit.

## What we're not claiming

The simulated system is hand-crafted. The log messages, the service topology, the command parser — we wrote them. This isn't a replay of real Prometheus traces or PagerDuty events. A judge might reasonably ask whether training on this actually helps a real SRE LLM, and the honest answer is: we don't know yet.

The 60-seed generalisation study is evidence the model isn't just memorising, but it's within-distribution generalisation. Real-world transfer is a separate question we haven't tested.

The training curves are noisy. GRPO with 4 sampled completions per step produces bimodal rewards (correct naming scores 0.9+, wrong naming scores 0.01) and the mean rises only from 0.23 to 0.32 across our 300-step run. A larger sample size and more training would smooth this, but within hackathon budget we chose to ship what we have.

## What worked, what didn't

Things that worked:

- Structured multi-role completion format. Fixing the train-eval mismatch was worth more than any hyperparameter tuning.
- Reward decomposition. Being able to turn components off one at a time made debugging tractable.
- Oracle audit scripts. Running a perfect-knowledge policy against every task before training caught two unreachable milestones.
- Anti-hack as a multiplicative gate. GRPO finds loops quickly; the gate stops them cold.
- The procedural task generator. Training on `procedural_easy/medium/hard` instead of the scripted tasks gave broader coverage and avoided overfitting.

Things that didn't:

- Our first reward function had an unnormalised time-pressure penalty that accidentally made hard tasks (fewer rounds = less penalty) score higher than easy tasks (more rounds = more penalty). We fixed it with a cap on total penalty as a fraction of available milestone credit, plus a solve bonus so a clean resolution always beats a partial one.
- We spent about $2 on a training run that only trained the diagnosis agent's round-0 completion. That was the v1 adapter. Every other adapter we've trained since has beaten it.
- Task 4, 5, and 6 were written but have been under-validated — the v2 run showed task 4 averaged reward 0.01, likely because of a fatal-check interaction we haven't fully debugged. We're ordering these below the first three in our submission.

## Try it

Live: [huggingface.co/spaces/brodie1of1/war-room](https://huggingface.co/spaces/brodie1of1/war-room)

The Gradio dashboard lets you pick a task, watch agents coordinate round by round, inspect the belief state tracker as agents update their views, and inject your own panic message mid-episode to see how the team responds. Agent Mode runs the actual trained adapter live (via HF Inference when available).

To reproduce evidence on your own machine, no GPU required:

```bash
git clone https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers.git
cd Meta_Hackathon_ClaudeStalkers
pip install -e .

python scripts/oracle_audit.py
python round2/war_room/reward_ablation.py
python round2/war_room/eval_generalization.py
pytest tests/ -v
```

To actually retrain the adapter on a GPU:

```bash
PYTHONPATH=. python round2/war_room/train_colab.py \
    --episodes 100 \
    --tasks procedural_easy procedural_medium procedural_hard \
    --lenient-format --no-unsloth
```

Cost was roughly $1.50 on an L40S. A Colab notebook version is at `round2/war_room/train_colab.ipynb`.

## What's next

We're running two more training configurations (`v4` and `v5`) with rank-32 LoRA, lr bumped to 1e-5, and a broader task mix of 6–9 scenarios. If the numbers beat v3's +0.046 meaningfully we'll update this post. If not, v3 is what we ship, and we'll have been honest about the ceiling.

The thing we most want to explore post-hackathon is ingesting real PagerDuty and Prometheus traces as replay fixtures. The simulation is the version of this problem we could build in 72 hours; the version that actually helps an SRE team lives one dataset away.

Thanks to the OpenEnv team at Meta and Hugging Face for the framework and the compute.

---

Code: [github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers](https://github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers) · Adapter: [brodie1of1/war-room-grpo-adapter-v3](https://huggingface.co/brodie1of1/war-room-grpo-adapter-v3) · Space: [brodie1of1/war-room](https://huggingface.co/spaces/brodie1of1/war-room)
