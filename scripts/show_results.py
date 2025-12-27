                                            
import json
from pathlib import Path

results_dir = Path('experiments/results')
files = list(results_dir.glob('full_comparison_*.json'))
latest = max(files, key=lambda p: p.name)

with open(latest) as f:
    data = json.load(f)

print('=' * 70)
print('EXPERIMENT RESULTS SUMMARY')
print('=' * 70)
print(f"Total experiments: {len(data['results'])}")
print()

results = [r for r in data['results'] if 'error' not in r]
sorted_results = sorted(results, key=lambda x: x['eval_reward_mean'], reverse=True)

print(f"{'Agent':<18} {'Reward':<20} {'Eval Mean':>10} {'Failures':>10} {'Entropy':>12}")
print('-' * 70)

for r in sorted_results:
    print(f"{r['agent']:<18} {r['reward']:<20} {r['eval_reward_mean']:>10.2f} "
          f"{r['failures_detected']:>10} {r['entropy_trend']:>12}")

print()
print('=' * 70)
print('TOP 5 BEST PERFORMERS')
print('=' * 70)
for i, r in enumerate(sorted_results[:5]):
    print(f"{i+1}. {r['agent']} + {r['reward']}: {r['eval_reward_mean']:.2f}")

print()
print('=' * 70)
print('BOTTOM 5 PERFORMERS')
print('=' * 70)
for i, r in enumerate(sorted_results[-5:]):
    print(f"{45-4+i}. {r['agent']} + {r['reward']}: {r['eval_reward_mean']:.2f}")

print()
print('=' * 70)
print('FAILURE ANALYSIS')
print('=' * 70)
failure_counts = {}
for r in results:
    for ftype in r.get('failure_types', []):
        failure_counts[ftype] = failure_counts.get(ftype, 0) + 1

if failure_counts:
    for ftype, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        print(f"  {ftype}: {count} experiments")
else:
    print("  No failures detected!")

print()
print('=' * 70)
print('AVERAGE PERFORMANCE BY AGENT')
print('=' * 70)

agents = set(r['agent'] for r in results)
for agent in sorted(agents):
    agent_results = [r for r in results if r['agent'] == agent]
    avg = sum(r['eval_reward_mean'] for r in agent_results) / len(agent_results)
    print(f"  {agent:<20}: {avg:.2f}")

print()
print('=' * 70)
print('AVERAGE PERFORMANCE BY REWARD')
print('=' * 70)

rewards = set(r['reward'] for r in results)
for reward in sorted(rewards):
    reward_results = [r for r in results if r['reward'] == reward]
    avg = sum(r['eval_reward_mean'] for r in reward_results) / len(reward_results)
    print(f"  {reward:<20}: {avg:.2f}")
