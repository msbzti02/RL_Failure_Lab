                                         
import json
from pathlib import Path

results_dir = Path('experiments/results')
files = list(results_dir.glob('full_comparison_*.json'))
latest = max(files, key=lambda p: p.name)

with open(latest) as f:
    data = json.load(f)

results = [r for r in data['results'] if 'error' not in r]
sorted_results = sorted(results, key=lambda x: x['eval_reward_mean'], reverse=True)

report = []
report.append("# RL Failure Lab - Experiment Results")
report.append("")
report.append(f"**Total experiments:** {len(results)}")
report.append(f"**Agents tested:** {len(set(r['agent'] for r in results))}")
report.append(f"**Reward functions tested:** {len(set(r['reward'] for r in results))}")
report.append("")

report.append("## Full Results Table")
report.append("")
report.append("| Rank | Agent | Reward | Eval Mean | Train Mean | Failures | Entropy Trend |")
report.append("|------|-------|--------|-----------|------------|----------|---------------|")

for i, r in enumerate(sorted_results):
    report.append(f"| {i+1} | {r['agent']} | {r['reward']} | {r['eval_reward_mean']:.2f} | {r['train_reward_mean']:.2f} | {r['failures_detected']} | {r['entropy_trend']} |")

report.append("")
report.append("## Top 5 Best Performers")
report.append("")
for i, r in enumerate(sorted_results[:5]):
    report.append(f"**{i+1}. {r['agent']} + {r['reward']}**")
    report.append(f"   - Eval reward: {r['eval_reward_mean']:.2f}")
    report.append(f"   - Train reward: {r['train_reward_mean']:.2f}")
    report.append(f"   - Failures: {r['failures_detected']}")
    report.append("")

report.append("## Bottom 5 Performers")
report.append("")
for i, r in enumerate(sorted_results[-5:]):
    report.append(f"**{41+i}. {r['agent']} + {r['reward']}**")
    report.append(f"   - Eval reward: {r['eval_reward_mean']:.2f}")
    report.append(f"   - Train reward: {r['train_reward_mean']:.2f}")
    report.append(f"   - Failures: {r['failures_detected']}")
    report.append("")

report.append("## Failure Analysis")
report.append("")
failure_counts = {}
for r in results:
    for ftype in r.get('failure_types', []):
        failure_counts[ftype] = failure_counts.get(ftype, 0) + 1

if failure_counts:
    report.append("| Failure Type | Experiments Affected |")
    report.append("|--------------|---------------------|")
    for ftype, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
        report.append(f"| {ftype} | {count} |")
else:
    report.append("No failures detected across all experiments.")

report.append("")
report.append("## Average Performance by Agent")
report.append("")
report.append("| Agent | Avg Eval Reward |")
report.append("|-------|-----------------|")

agents = set(r['agent'] for r in results)
agent_avgs = []
for agent in agents:
    agent_results = [r for r in results if r['agent'] == agent]
    avg = sum(r['eval_reward_mean'] for r in agent_results) / len(agent_results)
    agent_avgs.append((agent, avg))

for agent, avg in sorted(agent_avgs, key=lambda x: -x[1]):
    report.append(f"| {agent} | {avg:.2f} |")

report.append("")
report.append("## Average Performance by Reward Function")
report.append("")
report.append("| Reward | Avg Eval Reward |")
report.append("|--------|-----------------|")

rewards = set(r['reward'] for r in results)
reward_avgs = []
for reward in rewards:
    reward_results = [r for r in results if r['reward'] == reward]
    avg = sum(r['eval_reward_mean'] for r in reward_results) / len(reward_results)
    reward_avgs.append((reward, avg))

for reward, avg in sorted(reward_avgs, key=lambda x: -x[1]):
    report.append(f"| {reward} | {avg:.2f} |")

output_path = Path('experiments/RESULTS_REPORT.md')
output_path.write_text('\n'.join(report))
print(f"Report saved to: {output_path}")

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)

print(f"\nBest agent overall: {sorted_results[0]['agent']}")
print(f"Best reward function: {sorted_results[0]['reward']}")
print(f"Best combination score: {sorted_results[0]['eval_reward_mean']:.2f}")

print(f"\nWorst combination: {sorted_results[-1]['agent']} + {sorted_results[-1]['reward']}")
print(f"Worst score: {sorted_results[-1]['eval_reward_mean']:.2f}")

if failure_counts:
    print(f"\nMost common failure: {max(failure_counts, key=failure_counts.get)}")
    print(f"Total experiments with failures: {sum(1 for r in results if r['failures_detected'] > 0)}")
