"""Quick inspector for a GRPO rollout audit JSONL log."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else (
    '/Users/LOSATHIS2601/.cache/huggingface/hub/'
    'models--brodie1of1--war-room-grpo-adapter/snapshots/'
    '790f3dea576e4e804497fec8d38f561dd1a350e8/rollout_audit.jsonl'
)
with open(path) as f:
    lines = f.readlines()
print(f'total audit entries: {len(lines)}')
print()
print('--- first 3 ---')
for line in lines[:3]:
    e = json.loads(line)
    preview = e['completion_preview'][:220]
    print(f'step={e["step"]} task={e["task_id"]} env={e["env_reward"]:.3f}'
          f' fmt={e["format"]:.2f} comm={e["communication"]:.2f} anti={e["anti_hack"]:.2f}')
    print(f'   completion: {preview!r}')
    print()
print('--- last 3 ---')
for line in lines[-3:]:
    e = json.loads(line)
    preview = e['completion_preview'][:220]
    print(f'step={e["step"]} task={e["task_id"]} env={e["env_reward"]:.3f}'
          f' fmt={e["format"]:.2f} comm={e["communication"]:.2f} anti={e["anti_hack"]:.2f}')
    print(f'   completion: {preview!r}')
    print()
print('--- env_reward distribution ---')
rewards = [json.loads(l)['env_reward'] for l in lines]
import statistics
print(f'n={len(rewards)} min={min(rewards):.3f} max={max(rewards):.3f} '
      f'mean={statistics.mean(rewards):.3f} stdev={statistics.stdev(rewards) if len(rewards)>1 else 0:.3f}')
# By task
from collections import defaultdict
per_task = defaultdict(list)
for l in lines:
    e = json.loads(l)
    per_task[e['task_id']].append(e['env_reward'])
for t, rs in sorted(per_task.items()):
    print(f'  {t}: n={len(rs)} mean={statistics.mean(rs):.3f} max={max(rs):.3f}')
