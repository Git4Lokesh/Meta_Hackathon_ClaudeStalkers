# Demo video storyboard (90-120 seconds)

**Purpose**: Optional supplement to the blog post — a fast, visceral demonstration of what our submission actually does. Per hackathon submission requirements, a YouTube video is ALT to the blog, not a requirement. We already have the blog. This is upside.

**Target length**: 90-120 seconds. If you run over 120s, cut shot 2b.

**Recording tool**: QuickTime + screen recording + a USB mic. OBS works too if you're comfortable with it.

**Who records**: Lakshminath or Siddharth. ~45 minutes total (record + edit + upload).

---

## Shot 1: The hook (15 seconds)

**Visual**: Full-screen terminal with a scrolling Slack-style incident channel. You can stage this with a simple script that prints messages like:
```
[03:14:07] CEO: WHY IS THE SITE DOWN? FIX IT NOW
[03:14:08] monitoring: Redis memory at 72% critical
[03:14:09] triage-bot: Redis memory critical — restart Redis?
[03:14:10] diagnosis-bot: wait — Redis logs show it's fine
[03:14:11] diagnosis-bot: real issue is a DB password in database.yml
```

**Voiceover** (read this verbatim):

> "When production goes down at 3 AM, the loudest voice is often wrong. The dashboards are stale, the alerts misfire, and someone on the team has to push back with evidence. We trained an LLM to be the voice that pushes back."

**Cut**: hard cut to shot 2.

---

## Shot 2: The environment (30 seconds)

**Visual**: Your browser at `https://huggingface.co/spaces/brodie1of1/war-room`. The Gradio app is open to Task 3 (Cascading Failure with Conflicting Information).

**Action**:
1. (5s) Reset to Task 3 seed 42. Show the dashboard with phantom Redis alerts.
2. (10s) Point at each panel: Triage observation, Diagnosis observation, Remediation observation. Mention the strict permissions (Remediation can't read logs, Diagnosis can't restart, etc.).
3. (10s) Hit Play (or step through rounds 1-3). Show the agents coordinating through messages. The phantom Redis alert is visible on the triage panel.
4. (5s) Pause on the Belief State Tracker panel — point at the "phantom chase" vs "phantom detection" counter.

**Voiceover** (read over action):

> "Three agents share a channel but see different slices of the system. Triage reads alerts. Diagnosis reads logs. Remediation restarts services. None of them can solve the incident alone. And our environment injects phantom alerts — false information that tries to pull the team off-track. Every episode, a Belief State Tracker records whether agents update their beliefs based on evidence, or just follow whoever shouts loudest."

---

## Shot 2b (optional, 15 seconds): The reward

**Visual**: Same Space, open the "Reward Inspector" tab.

**Voiceover**:

> "Four independent reward signals — milestone, format, communication quality, anti-hack — that we can ablate one at a time to prove each component is earning its weight."

Cut this shot if your total is running over 105 seconds.

---

## Shot 3: The training result (30 seconds)

**Visual**: Switch to a split-screen layout — left side shows a real rollout of base Qwen 7B on Task 2 seed 33, right side shows the same scenario with our trained adapter. You can stage this offline by screenshotting the terminal output from `outputs/worked_example/task2_seed33_rollout.json` as ASCII, or record `python scripts/show_worked_example.py` output.

Alternative if time-constrained: show the head-to-head chart image from `outputs/llm_eval/v3/head_to_head.png` full-screen with annotations.

**Voiceover** (for chart version):

> "We trained Qwen 2.5-7B with GRPO on a single Hugging Face L40S job. Before: composite score 0.27. After our adapter: 0.32. On the memory leak task specifically — where the base model gets distracted by a CPU red herring — the score quadruples from 0.05 to 0.19. The trained model learned to not get fooled."

**Voiceover** (for split-screen rollout version):

> "Left side is base Qwen 7B. Right side is our trained adapter. Same incident, same seed, same prompts. The base model drifts onto the wrong service by round five. The trained model stays focused. That directed persistence is what gets the score from 0.05 to 0.19 on this task."

---

## Shot 4: Close (15 seconds)

**Visual**: A clean title card with:

```
Multi-Agent Incident War Room
Theme #1: Multi-Agent Interactions

Try it   → huggingface.co/spaces/brodie1of1/war-room
Adapter  → huggingface.co/brodie1of1/war-room-grpo-adapter-v3
Code     → github.com/Git4Lokesh/Meta_Hackathon_ClaudeStalkers

Team ClaudeStalkers — Siddharth, Lakshminath, Lokesh
BITS Pilani Hyderabad
```

**Voiceover**:

> "Live demo, trained adapter, and source code at the links on screen. Team ClaudeStalkers. Thanks for watching."

---

## Technical notes for recording

- **Audio**: Record in a quiet room. Laptop mics are OK. USB mic is better.
- **Resolution**: 1080p minimum. 1440p or 4K looks cleaner.
- **Screen recording**: QuickTime (Cmd+Shift+5 on Mac) captures the full screen or a window. OBS is free and more powerful.
- **Editing**: iMovie / DaVinci Resolve (free) for basic cuts. You only need to:
  1. Trim dead space at the start/end of each shot
  2. Stitch the 4 shots together with hard cuts (no fancy transitions)
  3. Ensure the voiceover audio syncs with the visuals
- **Export**: H.264 MP4, 1080p, ~10 MB per minute is fine.
- **Upload**: YouTube unlisted. Copy the URL. Add it to the README top block as `[🎬 Demo video](URL)`.

## Fallback if you can't record

The blog already satisfies the minimum submission requirement. The video is upside. If recording is blocking submission, ship without it.
