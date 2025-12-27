                          
import json
from pathlib import Path

results_dir = Path('experiments/results')
files = list(results_dir.glob('full_comparison_*.json'))
latest = max(files, key=lambda p: p.name)

with open(latest) as f:
    data = json.load(f)

results = [r for r in data['results'] if 'error' not in r]
sorted_results = sorted(results, key=lambda x: x['eval_reward_mean'], reverse=True)

print('TOP 10 PERFORMERS:')
print('-' * 60)
for i, r in enumerate(sorted_results[:10]):
    agent = r['agent']
    reward = r['reward']
    score = r['eval_reward_mean']
    print(f"{i+1:2}. {agent:18} + {reward:22} = {score:7.2f}")

print()
print('BOTTOM 5 PERFORMERS:')
print('-' * 60)
for i, r in enumerate(sorted_results[-5:]):
    agent = r['agent']
    reward = r['reward']
    score = r['eval_reward_mean']
    print(f"{41+i:2}. {agent:18} + {reward:22} = {score:7.2f}")
