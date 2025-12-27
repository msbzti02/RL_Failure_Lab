   
import sys
sys.path.insert(0, '.')

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from src.environments import CareerEnv
from src.environments.param_engine import EnvironmentParams
from src.rewards import get_reward, list_rewards
from src.agents import DQNAgent, PPOAgent, RandomAgent
from src.agents.heuristic_agents import create_heuristic_agent, HEURISTIC_AGENTS
from src.failure_detection import (
    CombinedFailureDetector,
    RewardHackingDetector,
    PolicyCollapseDetector,
    NonstationarityDetector,
    ValueExplosionDetector,
)
from src.introspection import EntropyTracker
from src.utils.reproducibility import set_global_seed

N_EPISODES = 100                                      
EVAL_EPISODES = 10
SEED = 42

AGENTS = [
    "dqn",
    "ppo",
    "random",
    "always_stay",
    "always_switch",
    "burnout_threshold",
    "conservative",
    "aggressive",
    "adaptive",
]

REWARDS = [
    "short_term_v1",
    "short_term_v2",
    "long_term_shaped_v1",
    "risk_sensitive_v1",
    "sparse_v1",
]

def create_agent(agent_type: str, state_dim: int, action_dim: int):
                                  
    if agent_type == "dqn":
        return DQNAgent(
            state_dim, action_dim,
            learning_rate=0.001,
            epsilon_decay=0.995,
            buffer_size=5000,
            batch_size=32
        )
    elif agent_type == "ppo":
        return PPOAgent(
            state_dim, action_dim,
            learning_rate=0.0003,
            n_steps=256,
            n_epochs=4
        )
    elif agent_type == "random":
        return RandomAgent(state_dim, action_dim)
    elif agent_type in HEURISTIC_AGENTS:
        return create_heuristic_agent(agent_type, state_dim, action_dim)
    else:
        raise ValueError(f"Unknown agent: {agent_type}")

def run_single_experiment(agent_type: str, reward_type: str, env_params=None):
                                                     
    set_global_seed(SEED)
    
    params = env_params or EnvironmentParams()
    env = CareerEnv(params)
    agent = create_agent(agent_type, 4, 2)
    reward_fn = get_reward(reward_type)
    
    detector = CombinedFailureDetector([
        RewardHackingDetector(window_size=50, min_correlation_samples=30),
        PolicyCollapseDetector(min_samples=50),
        NonstationarityDetector(min_samples=30),
        ValueExplosionDetector(min_samples=20),
    ])
    
    entropy_tracker = EntropyTracker(window_size=50)
    episode_rewards = []
    episode_lengths = []
    final_burnouts = []
    detected_failures = []
    
    for episode in range(N_EPISODES):
        state, _ = env.reset()
        reward_fn.reset()
        
        ep_reward = 0
        ep_steps = 0
        done = False
        
        while not done:
            action = agent.select_action(state, explore=True)
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            reward = reward_fn.compute(state, action, next_state, info, terminated, truncated)
            ep_reward += reward
            ep_steps += 1
            
            if hasattr(agent, 'update') and agent_type in ['dqn', 'ppo']:
                agent.update({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done
                })
            
            state = next_state
        
        if hasattr(agent, 'on_episode_end'):
            agent.on_episode_end()
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        final_burnouts.append(info.get('burnout', 0))
        
        entropy = agent.get_policy_entropy() if hasattr(agent, 'get_policy_entropy') else 0.5
        entropy_tracker.update(entropy, episode)
        
        detector.update({
            "step": episode,
            "episode": episode,
            "reward": ep_reward,
            "burnout": info.get('burnout', 0),
            "entropy": entropy,
            "action": action,
        })
        
        failures = detector.detect_all()
        for f in failures:
            detected_failures.append({
                "episode": episode,
                "type": f.failure_type.value,
                "severity": f.severity,
            })
    
    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        state, _ = env.reset()
        reward_fn.reset()
        ep_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, explore=False)
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            reward = reward_fn.compute(state, action, next_state, info, terminated, truncated)
            ep_reward += reward
            state = next_state
        
        eval_rewards.append(ep_reward)
    
    return {
        "agent": agent_type,
        "reward": reward_type,
        "train_reward_mean": float(np.mean(episode_rewards)),
        "train_reward_std": float(np.std(episode_rewards)),
        "train_reward_final": float(np.mean(episode_rewards[-20:])),
        "eval_reward_mean": float(np.mean(eval_rewards)),
        "eval_reward_std": float(np.std(eval_rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "avg_final_burnout": float(np.mean(final_burnouts)),
        "failures_detected": len(detected_failures),
        "failure_types": list(set(f["type"] for f in detected_failures)),
        "entropy_trend": entropy_tracker.get_trend(),
    }

def main():
    print("=" * 70)
    print("RL FAILURE LAB — COMPREHENSIVE EXPERIMENT SUITE")
    print("=" * 70)
    print(f"\nRunning {len(AGENTS)} agents x {len(REWARDS)} rewards = {len(AGENTS) * len(REWARDS)} experiments")
    print(f"Episodes per experiment: {N_EPISODES}")
    print(f"Evaluation episodes: {EVAL_EPISODES}")
    print()
    
    all_results = []
    
    total = len(AGENTS) * len(REWARDS)
    pbar = tqdm(total=total, desc="Running experiments")
    
    for agent_type in AGENTS:
        for reward_type in REWARDS:
            try:
                result = run_single_experiment(agent_type, reward_type)
                all_results.append(result)
                
                pbar.set_postfix({
                    "agent": agent_type[:8],
                    "reward": reward_type[:12],
                    "eval": f"{result['eval_reward_mean']:.1f}"
                })
            except Exception as e:
                print(f"\nError with {agent_type} + {reward_type}: {e}")
                all_results.append({
                    "agent": agent_type,
                    "reward": reward_type,
                    "error": str(e),
                })
            
            pbar.update(1)
    
    pbar.close()
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Agent':<18} {'Reward':<20} {'Eval Mean':>10} {'Failures':>10} {'Entropy':>12}")
    print("-" * 70)
    
    valid_results = [r for r in all_results if 'error' not in r]
    sorted_results = sorted(valid_results, key=lambda x: x['eval_reward_mean'], reverse=True)
    
    for r in sorted_results:
        print(f"{r['agent']:<18} {r['reward']:<20} {r['eval_reward_mean']:>10.2f} "
              f"{r['failures_detected']:>10} {r['entropy_trend']:>12}")
    
    print("\n" + "=" * 70)
    print("TOP PERFORMERS BY REWARD FUNCTION")
    print("=" * 70)
    
    for reward_type in REWARDS:
        reward_results = [r for r in valid_results if r['reward'] == reward_type]
        if reward_results:
            best = max(reward_results, key=lambda x: x['eval_reward_mean'])
            print(f"{reward_type:<25} Best: {best['agent']:<15} (eval={best['eval_reward_mean']:.2f})")
    
    print("\n" + "=" * 70)
    print("TOP PERFORMERS BY AGENT")
    print("=" * 70)
    
    for agent_type in AGENTS:
        agent_results = [r for r in valid_results if r['agent'] == agent_type]
        if agent_results:
            best = max(agent_results, key=lambda x: x['eval_reward_mean'])
            avg = np.mean([r['eval_reward_mean'] for r in agent_results])
            print(f"{agent_type:<18} Best reward: {best['reward']:<20} Avg: {avg:.2f}")
    
    print("\n" + "=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)
    
    failure_counts = {}
    for r in valid_results:
        for ftype in r.get('failure_types', []):
            failure_counts[ftype] = failure_counts.get(ftype, 0) + 1
    
    if failure_counts:
        for ftype, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
            print(f"  {ftype}: {count} experiments")
    else:
        print("  No failures detected!")
    
    output_dir = Path("experiments/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"full_comparison_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "n_episodes": N_EPISODES,
                "eval_episodes": EVAL_EPISODES,
                "seed": SEED,
                "agents": AGENTS,
                "rewards": REWARDS,
            },
            "results": all_results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("EXPERIMENT SUITE COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
