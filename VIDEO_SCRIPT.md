# 2-minute video script — Multi-Agent Incident War Room

**Target length**: 1:50–2:00
**Target audience**: hackathon judge who has 90 seconds to decide if this submission is worth a longer look.
**Tone**: first-person engineering, quiet confidence, no marketing speak. Natural cadence — read it like you're explaining to a colleague.
**Recording tool**: Loom / QuickTime / OBS with screen capture at 1080p. Voiceover over the Gradio dashboard + committed charts.

---

## 0:00–0:15 — Hook and the problem (15s)

**[Screen: the Gradio dashboard with Task 3 selected, pre-reset]**

> "At 3 AM during an outage, the loneliest part isn't the alert. It's the moment someone in Slack says with full confidence 'it's the database,' and the logs quietly suggest it isn't. We built an OpenEnv environment where three AI agents have to resolve a production incident, and every third round a simulated executive panics at them. We train Qwen 7B not to get fooled."

---

## 0:15–0:45 — Show the base model getting fooled (30s)

**[Action: on screen, check Agent Mode, confirm "Base Qwen 7B" preset selected, click Start, then Next a few times on Task 3]**

> "Task 3 is the hard one. Dashboard shows Redis memory at 72 percent, critical alert. The real root cause is a wrong database password. The Redis alert is stale."

**[Wait for agents to respond. Point at the output when base model issues a bad command]**

> "Watch base Qwen here — Triage panics at the Redis alert, Diagnosis restarts a healthy Redis, and the actual database credential bug stays broken for the whole episode. That's the failure mode we're targeting."

**[Show belief-state tracker panel updating as rounds go by]**

> "The Belief State Tracker on the right records whose beliefs moved on evidence versus whose moved on panic. That's the signal we train against."

---

## 0:45–1:15 — Show the reward design and the eval delta (30s)

**[Cut to: README scrolled to the reward table + ablation chart]**

> "Four independent reward functions. Milestone. Format. Communication. Anti-hack as a multiplicative gate — if the model loops the same command three times, the whole reward zeros out. We ablate each component on fixed seeds to show every signal earns its weight."

**[Cut to: outputs/llm_eval/v3/head_to_head.png full screen]**

> "We trained Qwen 7B with GRPO plus LoRA on a single L40S. Our v3 adapter beats base Qwen 7B by 4.6 percentage points on composite score, with a 4× lift on the memory-leak task. First adapter in our iteration series that landed on the right side of zero — v1 and v2 actually did worse than base, and we documented exactly why."

---

## 1:15–1:45 — Show the worked example + honest progression (30s)

**[Cut to: Blog.md 30-second teaser section — the verbatim base-vs-trained rollout quote at the top]**

> "This is the exact rollout we ship in the blog. Task 2, memory leak on data_processor, CPU red herring on api_gateway. By round 5, base Qwen has drifted onto the CPU track. Same model family, same prompts, only the LoRA differs — the trained model keeps remediation pointed at the real fault. Small win on paper, but it's the model learning to resist the kind of distraction that makes 3 AM incidents drag on for hours."

**[Cut to: outputs/v6_vs_v7_comparison/comparison_charts.png]**

> "Our most recent training run — v7 — landed a reward-function fix during the onsite. Four constants changed in grader.py. Task 2, 3, 5, and 6 were stuck at the reward floor across 800 episodes of v5 because the penalty math was wrong. v7 sits in the partial-credit zone with barely any floor pile-up. Head-to-head eval is running as we submit."

---

## 1:45–2:00 — The wrap (15s)

**[Cut to: Space URL and Colab link visible on screen]**

> "Everything is in the Hugging Face Space — the Gradio dashboard, the training notebook, the blog with the raw rollout traces, and a full run log with every adapter we've trained, including the ones that failed. Thanks."

---

## Recording notes

**Visuals to prepare before recording:**
1. Gradio dashboard loaded with Task 3 selected, Agent Mode unchecked.
2. Browser tab 2: README.md scrolled to the reward table + ablation chart.
3. Browser tab 3: `outputs/llm_eval/v3/head_to_head.png` opened full-screen.
4. Browser tab 4: Blog.md scrolled to the 30-second teaser section.
5. Browser tab 5: `outputs/v6_vs_v7_comparison/comparison_charts.png` opened full-screen.

**Cadence tips:**
- Don't rush. 2 minutes is enough for this script at a conversational pace.
- Pause briefly between sections to let the visual land.
- No upbeat music. Keep it engineering-podcast flat.

**After recording:**
1. Upload to YouTube as **unlisted** (or public if team is comfortable).
2. Copy the URL.
3. Add to README top — one line under the hero chart:
   `📺 [Watch a 2-minute walkthrough on YouTube](https://youtu.be/...)`
4. Add to SUBMISSION_CHECKLIST.md under "Rubric Strongest-Evidence Pointers → Storytelling".
5. Re-run `deploy_hf_space.py` so the Space README shows the video link.

**If time runs tight:**
- The 1:15–1:45 worked-example section is the second-most-dispensable. You can cut it to 15s.
- The 1:45–2:00 wrap can compress to 5s.
- The 0:45–1:15 reward-and-eval section is the one that *cannot* be cut — that's the rubric evidence.

**Fallback if recording quality is poor:**
Upload anyway. Per the organizer's rubric, a present-but-rough video is worth more than no video. Imperfect shipped beats perfect unshipped.

---

## Why this script works against the rubric

| Section | Seconds | Rubric dimension moved |
|---|---|---|
| Hook (0:00–0:15) | 15 | Storytelling (problem framing) |
| Base model fooled (0:15–0:45) | 30 | Environment Innovation (shows the phantom-alert mechanic works) |
| Rewards + eval delta (0:45–1:15) | 30 | Improvement + Reward/Pipeline (rubric evidence) |
| Worked example + v7 (1:15–1:45) | 30 | Storytelling (honest progression) + Improvement (v7 signal) |
| Wrap (1:45–2:00) | 15 | Storytelling (call to action) |

This order follows the hackathon guide's explicit rubric-weight order: Environment (40%) gets the most vivid demo, Storytelling (30%) gets the most prose, Improvement (20%) gets the chart, Pipeline (10%) gets mentioned in passing.
