# 90-Second Demo Video — Script + Storyboard

Target: <2 minutes per the hackathon minimum submission requirements. Record in one take if possible, split into two if not. Upload unlisted to YouTube, link from README.

**Setup before recording:**

- Open the HF Space in a browser tab: https://brodie1of1-war-room.hf.space
- Open a second tab with the README rendered on GitHub
- QuickTime screen recorder at 1920×1080 or 1440×900, system audio ON
- Disable notifications
- Have a glass of water

---

## Storyboard (90 seconds total)

### [0:00 – 0:10] Hook
**Visual**: Open the HF Space. Zoom on the header "Multi-Agent Incident War Room".

**Voiceover**:
> "Production incidents are never solved by one person. We built an environment that teaches AI agents to diagnose them together under deception."

### [0:10 – 0:25] The setup
**Visual**: Click the Task dropdown → select **Task 3 (Cascading Failure with Conflicting Info)**. Point at the three agent columns in Gradio.

**Voiceover**:
> "Three agents with strict partial observability. Triage sees the dashboard. Diagnosis reads logs. Remediation restarts services. They share only a message board."

### [0:25 – 0:40] Phantom alert introduction
**Visual**: Click **Start**. Zoom the Belief State panel. Point at the phantom Redis alert on Triage's dashboard.

**Voiceover**:
> "Task 3 injects a phantom alert. Triage panics about Redis memory. But Redis is actually fine — the real problem is a wrong database password buried in the config."

### [0:40 – 1:05] Theory of Mind moment
**Visual**: Click **Next** three times. When the purple "🧠 Theory of Mind moment" banner fires in the chat, zoom on it and on the top-of-chat pushback summary banner.

**Voiceover**:
> "Here's the key moment. Diagnosis reads the Redis log, finds nothing wrong, and pushes back: 'Redis is fine — the real issue is db_connector auth.' That pushback is what we call a Theory of Mind moment. The environment counts them, and trained agents produce them reliably."

### [1:05 – 1:20] Proof
**Visual**: Switch to GitHub README tab. Scroll to **Training Curves** and **Generalization** sections. Zoom the two plots.

**Voiceover**:
> "We trained Qwen-7B with GRPO for 91 episodes on an L40S — costs a dollar ten. Composite score on scripted tasks: 0.01 to 0.80. On sixty unseen procedurally-generated scenarios, the baseline flatlines while the trained policy resolves 55% of the hardest ones."

### [1:20 – 1:30] Close
**Visual**: Switch to terminal. Run `PYTHONPATH=. python round2/war_room/demo_comparison.py`. Show the before/after table in under 1 second.

**Voiceover**:
> "Try it in 30 seconds: clone the repo, run `demo_comparison.py`, see for yourself. Everything's reproducible. Multi-agent cooperation under deception — link in the description."

---

## Recording checklist

- [ ] QuickTime > File > New Screen Recording
- [ ] Pick 1440×900 or 1920×1080
- [ ] System audio on
- [ ] Browser fullscreen (Cmd+Shift+F)
- [ ] Disable notifications (Focus mode)
- [ ] Run through the script once silently
- [ ] Record in one take, 2-3 attempts to get it right
- [ ] Export as MP4, max 50 MB
- [ ] Upload to YouTube as Unlisted
- [ ] Paste link into README under "Demo Video" row

## Backup

If the live demo glitches mid-record, open the scripted heuristic mode (Agent Mode off). It deterministically produces a Theory of Mind moment on Task 3 at round 3. Re-record that segment only and stitch if needed.

## Format notes

- Natural voice, no music. Judges are watching on small screens.
- Don't over-rehearse. A slightly nervous pitch reads more authentic than a polished one.
- Cover composite score 0.01 → 0.80 verbally — that's the number that sticks.
- End on the `demo_comparison.py` invocation so the judge thinks "I could run this myself."
