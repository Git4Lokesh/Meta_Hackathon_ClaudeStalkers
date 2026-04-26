"""Display the captured base vs trained rollout for the worked example."""
import json


def show():
    with open('outputs/worked_example/task2_seed33_rollout.json') as f:
        trace = json.load(f)

    for label in ['base', 'trained']:
        t = trace[label]
        print('=' * 70)
        print(f' {label.upper():7s} on {t["task"]} seed={t["seed"]}')
        print(f'   final_score={t["final_score"]}  rounds={t["total_rounds"]}')
        print(f'   milestones: {t["milestones"]}')
        print('=' * 70)
        for r in t['rounds'][:4]:
            print(f'\n  Round {r["round"]}  score_so_far={r["score_so_far"]}')
            print(f'  milestones_hit: {r["milestones_hit"]}')
            for role in ['triage', 'diagnosis', 'remediation']:
                rd = r['roles'][role]
                cmd = rd.get('parsed_command') or '(no command)'
                msg = rd.get('parsed_message') or ''
                print(f'    [{role:11s}] cmd:     {cmd[:120]!r}')
                if msg:
                    print(f'                  msg:     {msg[:180]!r}')
        print()


if __name__ == '__main__':
    show()
