   
import sys
sys.path.insert(0, '.')

from src.environments import CareerEnv
from src.environments.param_engine import EnvironmentParams
from src.rewards import get_reward, list_rewards
from src.agents import DQNAgent, RandomAgent
from src.agents.heuristic_agents import BurnoutThresholdAgent
from src.failure_detection import CombinedFailureDetector, PolicyCollapseDetector, RewardHackingDetector
from src.utils.reproducibility import set_global_seed

def demo_environment():
                                        
    print("\n" + "="*60)
    print("1. ENVIRONMENT DEMO")
    print("="*60)
    
    params = EnvironmentParams(
        burnout_rate=0.1,
        salary_volatility=0.15,
        market_regime="stable"
    )
    env = CareerEnv(params)
    
    print(f"Environment config: {env.get_config()}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset(seed=42)
    print(f"\nInitial state: {obs}")
    print(f"Initial info: {info}")
    
    print("\nRunning 5 random steps:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {i+1}: Action={'STAY' if action==0 else 'SWITCH'}, "
              f"Reward={reward:.3f}, Burnout={info['burnout']:.2f}")
        
        if terminated or truncated:
            break
    
    print(env.render(mode="ansi"))

def demo_rewards():
                                       
    print("\n" + "="*60)
    print("2. REWARD FUNCTIONS DEMO")
    print("="*60)
    
    print("Available reward functions:")
    for name in list_rewards():
        print(f"  - {name}")
    
    import numpy as np
    state = np.array([0.5, 0.3, 0.2, 0.0])                           
    next_state = np.array([0.6, 0.4, 0.25, 0.1])                                 
    info = {"salary": 80000, "burnout": 0.4}
    
    print("\nComparing rewards for same transition:")
    for reward_name in ["short_term_v1", "short_term_v2", "sparse_v1"]:
        reward_fn = get_reward(reward_name)
        r = reward_fn.compute(state, 0, next_state, info, False, False)
        print(f"  {reward_name}: {r:.3f}")

def demo_agents():
                                  
    print("\n" + "="*60)
    print("3. AGENTS DEMO")
    print("="*60)
    
    set_global_seed(42)
    
    env = CareerEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agents = {
        "DQN": DQNAgent(state_dim, action_dim, epsilon_decay=0.99),
        "Random": RandomAgent(state_dim, action_dim),
        "Burnout Threshold": BurnoutThresholdAgent(state_dim, action_dim, burnout_threshold=0.5),
    }
    
    print("\nTraining DQN for 20 episodes...")
    dqn = agents["DQN"]
    reward_fn = get_reward("short_term_v1")
    
    episode_rewards = []
    for episode in range(20):
        state, _ = env.reset()
        reward_fn.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = dqn.select_action(state, explore=True)
            next_state, base_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            reward = reward_fn.compute(state, action, next_state, info, terminated, truncated)
            total_reward += reward
            
            dqn.update({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            })
            
            state = next_state
        
        episode_rewards.append(total_reward)
        dqn.on_episode_end()
    
    print(f"  Episode rewards: {[f'{r:.1f}' for r in episode_rewards[-5:]]}")
    print(f"  Agent diagnostics: epsilon={dqn.epsilon:.3f}, updates={dqn._updates}")

def demo_failure_detection():
                                        
    print("\n" + "="*60)
    print("4. FAILURE DETECTION DEMO")
    print("="*60)
    
    detector = CombinedFailureDetector([
        PolicyCollapseDetector(entropy_threshold=0.1, dominance_threshold=0.9),
        RewardHackingDetector(),
    ])
    
    print("\nSimulating policy collapse scenario...")
    for i in range(150):
                                                          
        detector.update({
            "step": i,
            "episode": i,
            "entropy": 0.5 - i * 0.003,                      
            "action": 0,               
            "action_distribution": [0.99, 0.01],
            "reward": 0.5,
            "burnout": 0.3,
        })
    
    failures = detector.detect_all()
    for failure in failures:
        print(f"\n⚠️ DETECTED: {failure}")

def demo_quick_experiment():
                                   
    print("\n" + "="*60)
    print("5. QUICK EXPERIMENT DEMO")
    print("="*60)
    
    set_global_seed(42)
    
    env = CareerEnv()
    agent = DQNAgent(
        state_dim=4, 
        action_dim=2,
        epsilon_decay=0.995,
        buffer_size=1000,
        batch_size=32
    )
    reward_fn = get_reward("short_term_v1", lambda_burnout=1.0)
    
    print("\nRunning 50 episode experiment...")
    
    train_rewards = []
    for episode in range(50):
        state, _ = env.reset()
        reward_fn.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            reward = reward_fn.compute(state, action, next_state, info, terminated, truncated)
            total_reward += reward
            
            agent.update({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            })
            
            state = next_state
        
        train_rewards.append(total_reward)
        agent.on_episode_end()
        
        if (episode + 1) % 10 == 0:
            avg_reward = sum(train_rewards[-10:]) / 10
            print(f"  Episode {episode+1}: Avg Reward = {avg_reward:.2f}, "
                  f"Epsilon = {agent.epsilon:.3f}")
    
    print(f"\nFinal diagnostics:")
    diag = agent.get_diagnostics()
    print(f"  Total steps: {diag['total_steps']}")
    print(f"  Updates: {diag['updates']}")
    print(f"  Policy entropy: {diag['policy_entropy']:.4f}")

def main():
    print("="*60)
    print("RL FAILURE LAB — QUICK START DEMO")
    print("="*60)
    
    demo_environment()
    demo_rewards()
    demo_agents()
    demo_failure_detection()
    demo_quick_experiment()
    
    print("\n" + "="*60)
    print("✅ Demo complete! All components working.")
    print("="*60)
    print("\nNext steps:")
    print("  1. Read docs/api_reference.md for full API")
    print("  2. Read docs/failure_playbook.md for debugging")
    print("  3. Run experiments with ExperimentRunner")

if __name__ == "__main__":
    main()
