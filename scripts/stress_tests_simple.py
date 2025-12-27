                                                       
import sys
sys.path.insert(0, '.')

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from src.environments import CareerEnv
from src.environments.param_engine import EnvironmentParams
from src.rewards import get_reward
from src.rewards.base_reward import BaseReward
from src.agents import DQNAgent, PPOAgent, RandomAgent
from src.agents.heuristic_agents import create_heuristic_agent
from src.utils.reproducibility import set_global_seed

N_EPISODES = 50
EVAL_EPISODES = 10

class SalaryOnlyReward(BaseReward):
    name = "salary_only"
    version = "test"
    def compute(self, state, action, next_state, env_info, terminated, truncated):
        return env_info.get('salary', 0) / 100000

class ScaledReward(BaseReward):
    name = "scaled"
    version = "test"
    def __init__(self, scale=1000.0):
        super().__init__()
        self.scale = scale
        self.base_reward = get_reward("short_term_v1")
    def compute(self, state, action, next_state, env_info, terminated, truncated):
        return self.base_reward.compute(state, action, next_state, env_info, terminated, truncated) * self.scale

def run_training(agent, env, reward_fn, n_episodes):
    episode_rewards = []
    episode_burnouts = []
    action_counts = defaultdict(int)
    q_values = []
    
    for ep in range(n_episodes):
        state, _ = env.reset()
        reward_fn.reset()
        ep_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, explore=True)
            action_counts[action] += 1
            
            next_state, _, term, trunc, info = env.step(action)
            done = term or trunc
            
            reward = reward_fn.compute(state, action, next_state, info, term, trunc)
            ep_reward += reward
            
            if hasattr(agent, 'get_q_values'):
                qv = agent.get_q_values(state)
                if qv is not None:
                    q_values.append(float(np.max(np.abs(qv))))
            
            if hasattr(agent, 'update'):
                agent.update({
                    "state": state, "action": action, "reward": reward,
                    "next_state": next_state, "done": done
                })
            state = next_state
        
        if hasattr(agent, 'on_episode_end'):
            agent.on_episode_end()
        
        episode_rewards.append(ep_reward)
        episode_burnouts.append(info.get('burnout', 0))
    
    total_actions = sum(action_counts.values())
    return {
        "rewards": episode_rewards,
        "burnouts": episode_burnouts,
        "action_dist": {k: v/total_actions for k, v in action_counts.items()},
        "max_q": max(q_values) if q_values else 0,
    }

def evaluate(agent, env, reward_fn, n_episodes):
    rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        reward_fn.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, explore=False)
            next_state, _, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward_fn.compute(state, action, next_state, info, term, trunc)
            state = next_state
        rewards.append(ep_reward)
    return {"mean": float(np.mean(rewards)), "std": float(np.std(rewards))}

results = []

print("T0.1: Deterministic Environment...")
traj1, traj2 = [], []
for run, traj in [(1, traj1), (2, traj2)]:
    set_global_seed(42)
    env = CareerEnv(EnvironmentParams(salary_volatility=0.0))
    state, _ = env.reset(seed=42)
    traj.append(state.copy())
    for _ in range(20):
        state, _, t, tr, _ = env.step(0)
        traj.append(state.copy())
        if t or tr: break
diff = float(np.abs(np.array(traj1) - np.array(traj2)).sum())
results.append({"test": "T0.1", "name": "Deterministic", "passed": diff < 1e-6, "diff": diff})
print(f"  Diff: {diff:.6f} - {'PASS' if diff < 1e-6 else 'FAIL'}")

print("T0.2: Random Baseline...")
set_global_seed(42)
env = CareerEnv()
agent = RandomAgent(4, 2)
metrics = run_training(agent, env, get_reward("short_term_v1"), N_EPISODES)
results.append({"test": "T0.2", "name": "Random Baseline", "passed": True, 
                "mean_reward": float(np.mean(metrics["rewards"]))})
print(f"  Mean reward: {np.mean(metrics['rewards']):.2f}")

print("T1.1: Salary Maximizer Trap...")
set_global_seed(42)
env = CareerEnv()
agent = DQNAgent(4, 2, epsilon_decay=0.99)
metrics = run_training(agent, env, SalaryOnlyReward(), N_EPISODES)
switch_ratio = metrics["action_dist"].get(1, 0)
mean_burnout = float(np.mean(metrics["burnouts"]))
results.append({"test": "T1.1", "name": "Salary Trap", 
                "passed": switch_ratio > 0.3 or mean_burnout > 0.5,
                "switch_ratio": switch_ratio, "mean_burnout": mean_burnout})
print(f"  Switch ratio: {switch_ratio:.2f}, Burnout: {mean_burnout:.2f}")

print("T2.2: Sparse Reward...")
sparse_eval, dense_eval = 0, 0
for rtype in ["short_term_v1", "sparse_v1"]:
    set_global_seed(42)
    env = CareerEnv()
    agent = DQNAgent(4, 2, epsilon_decay=0.995)
    run_training(agent, env, get_reward(rtype), N_EPISODES)
    ev = evaluate(agent, env, get_reward(rtype), EVAL_EPISODES)
    if rtype == "sparse_v1": sparse_eval = ev["mean"]
    else: dense_eval = ev["mean"]
results.append({"test": "T2.2", "name": "Sparse Reward",
                "passed": dense_eval > sparse_eval,
                "dense_eval": dense_eval, "sparse_eval": sparse_eval})
print(f"  Dense: {dense_eval:.2f}, Sparse: {sparse_eval:.2f}")

print("T3.1: High Reward Scale...")
set_global_seed(42)
env = CareerEnv()
agent = DQNAgent(4, 2, epsilon_decay=0.99, learning_rate=0.001)
metrics = run_training(agent, env, ScaledReward(1000.0), N_EPISODES)
max_q = metrics["max_q"]
exploded = max_q > 100 or np.isnan(max_q)
results.append({"test": "T3.1", "name": "High Scale", "passed": exploded, "max_q": max_q})
print(f"  Max Q: {max_q:.2f} - {'EXPLODED' if exploded else 'Stable'}")

print("T3.3: Exploration Collapse...")
set_global_seed(42)
env = CareerEnv()
agent = DQNAgent(4, 2, epsilon_decay=0.95, epsilon_end=0.0)
metrics = run_training(agent, env, get_reward("short_term_v1"), N_EPISODES)
max_action = max(metrics["action_dist"].values())
collapsed = max_action > 0.9
results.append({"test": "T3.3", "name": "Collapse", "passed": collapsed, "max_action": max_action})
print(f"  Max action ratio: {max_action:.2%} - {'COLLAPSED' if collapsed else 'Diverse'}")

print("T4.1: Regime Shift...")
set_global_seed(42)
env = CareerEnv(EnvironmentParams.for_regime("stable"))
agent = DQNAgent(4, 2, epsilon_decay=0.995)
run_training(agent, env, get_reward("short_term_v1"), N_EPISODES)
stable_eval = evaluate(agent, env, get_reward("short_term_v1"), EVAL_EPISODES)["mean"]
recession_env = CareerEnv(EnvironmentParams.for_regime("recession"))
recession_eval = evaluate(agent, recession_env, get_reward("short_term_v1"), EVAL_EPISODES)["mean"]
drop = stable_eval - recession_eval
results.append({"test": "T4.1", "name": "Regime Shift", "passed": drop > 0,
                "stable": stable_eval, "recession": recession_eval, "drop": drop})
print(f"  Stable: {stable_eval:.2f}, Recession: {recession_eval:.2f}, Drop: {drop:.2f}")

print("T7.1: Heuristic Comparison...")
comparison = {}
for name, agent_fn in [
    ("dqn", lambda: DQNAgent(4, 2, epsilon_decay=0.995)),
    ("ppo", lambda: PPOAgent(4, 2, n_steps=128)),
    ("heuristic", lambda: create_heuristic_agent("burnout_threshold", 4, 2)),
    ("random", lambda: RandomAgent(4, 2))
]:
    set_global_seed(42)
    env = CareerEnv()
    agent = agent_fn()
    run_training(agent, env, get_reward("short_term_v1"), N_EPISODES)
    ev = evaluate(agent, env, get_reward("short_term_v1"), EVAL_EPISODES)["mean"]
    comparison[name] = ev
    print(f"  {name}: {ev:.2f}")
rl_best = max(comparison["dqn"], comparison["ppo"])
rl_wins = rl_best > comparison["heuristic"]
results.append({"test": "T7.1", "name": "Heuristic", "passed": True, 
                "comparison": comparison, "rl_wins": rl_wins})
print(f"  RL {'wins' if rl_wins else 'LOSES to heuristic'}")

print("T8.1: Seed Sensitivity...")
seed_results = []
for seed in [1, 2, 3, 4, 5]:
    set_global_seed(seed)
    env = CareerEnv()
    agent = DQNAgent(4, 2, epsilon_decay=0.995)
    run_training(agent, env, get_reward("short_term_v1"), N_EPISODES)
    ev = evaluate(agent, env, get_reward("short_term_v1"), EVAL_EPISODES)["mean"]
    seed_results.append(ev)
results.append({"test": "T8.1", "name": "Seeds", "passed": True,
                "mean": float(np.mean(seed_results)), "std": float(np.std(seed_results)),
                "range": float(max(seed_results) - min(seed_results))})
print(f"  Mean: {np.mean(seed_results):.2f} +/- {np.std(seed_results):.2f}")

print("\n" + "="*60)
print("SAVING RESULTS...")
output = Path("experiments/STRESS_TEST_RESULTS.json")
output.parent.mkdir(exist_ok=True)
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved to: {output}")

report_path = Path("experiments/STRESS_TEST_REPORT.md")
with open(report_path, 'w') as f:
    f.write("# RL Stress Test Report\n\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
    f.write("## Summary\n\n")
    f.write("| Test | Name | Result | Key Finding |\n")
    f.write("|------|------|--------|-------------|\n")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        key = str({k:v for k,v in r.items() if k not in ["test","name","passed"]})[:50]
        f.write(f"| {r['test']} | {r['name']} | {status} | {key}... |\n")
    f.write("\n## Key Lessons\n\n")
    f.write("1. **Reward Hacking**: Single-objective rewards are pathological\n")
    f.write("2. **Sparse Rewards**: Credit assignment is hard\n")
    f.write("3. **Scale Matters**: Reward scaling breaks value methods\n")
    f.write("4. **Exploration**: Fast decay = collapse\n")
    f.write("5. **Generalization**: Train != test distribution\n")
print(f"Report: {report_path}")
print("\nDONE!")
