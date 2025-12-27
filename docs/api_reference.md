# RL Failure Lab — API Reference

> Complete API documentation for the RL experimentation framework.

## Environments

### CareerEnv

A career simulation environment for studying sequential decision-making.

```python
from src.environments import CareerEnv
from src.environments.param_engine import EnvironmentParams

# Create with default params
env = CareerEnv()

# Create with custom params
params = EnvironmentParams(
    burnout_rate=0.1,
    salary_volatility=0.15,
    switching_risk=0.3,
    market_regime="stable"  # stable, boom, recession, volatile
)
env = CareerEnv(params)
```

#### State Space
- **Dimension:** 4
- **Components:**
  - `salary_norm`: Normalized salary [0, 1]
  - `burnout`: Current burnout level [0, 1]
  - `tenure_norm`: Normalized tenure [0, 1]
  - `market_condition`: Market state [-1, 1]

#### Action Space
- **Type:** Discrete(2)
- **Actions:**
  - `0`: STAY at current job
  - `1`: SWITCH jobs

#### Methods

```python
# Standard Gymnasium interface
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step(action)

# Custom methods
config = env.get_config()  # Get full configuration
summary = env.get_episode_summary()  # Episode statistics
env.render(mode="human")  # Visualize state
```

### EnvironmentParams

Configuration dataclass for environment parameters.

```python
from src.environments.param_engine import EnvironmentParams, MARKET_REGIMES

# List available regimes
print(MARKET_REGIMES.keys())  # ['stable', 'boom', 'recession', 'volatile']

# Create params for specific regime
params = EnvironmentParams.for_regime("recession")

# Validate parameters
params._validate()  # Raises ValueError if invalid

# Convert to dictionary
config_dict = params.to_dict()
```

---

## Reward Functions

### Registry

```python
from src.rewards import get_reward, list_rewards, RewardRegistry

# List available rewards
print(list_rewards())
# ['short_term_v1', 'short_term_v2', 'long_term_shaped_v1', ...]

# Get a reward function
reward_fn = get_reward("short_term_v1", lambda_burnout=1.5)

# Use in training loop
reward = reward_fn.compute(state, action, next_state, info, terminated, truncated)
reward_fn.reset()  # Call at episode start
```

### Available Reward Functions

| Name | Description |
|------|-------------|
| `short_term_v1` | `salary - λ * burnout` |
| `short_term_v2` | `salary - λ * burnout²` (non-linear) |
| `long_term_shaped_v1` | Potential-based shaping |
| `long_term_shaped_v2` | Shaping with momentum |
| `risk_sensitive_v1` | Variance-penalized |
| `risk_sensitive_cvar_v1` | CVaR-based |
| `sparse_v1` | Goal-only reward |
| `sparse_milestones_v1` | Milestone bonuses |
| `delayed_v1` | End-of-episode only |
| `delayed_hindsight_v1` | Hindsight analysis |

### Creating Custom Rewards

```python
from src.rewards.base_reward import BaseReward
from src.rewards.registry import register_reward

@register_reward("my_reward_v1")
class MyReward(BaseReward):
    name = "my_reward"
    version = "v1"
    description = "Custom reward function"
    
    def compute(self, state, action, next_state, env_info, terminated, truncated):
        # Your logic here
        return reward_value
```

---

## Agents

### DQNAgent

Deep Q-Network with experience replay and target network.

```python
from src.agents import DQNAgent

agent = DQNAgent(
    state_dim=4,
    action_dim=2,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=64,
    hidden_dims=[64, 64],
    target_update_freq=100,
)

# Training loop
action = agent.select_action(state, explore=True)
metrics = agent.update({
    "state": state,
    "action": action,
    "reward": reward,
    "next_state": next_state,
    "done": done
})

# Diagnostics
print(agent.get_diagnostics())
print(agent.get_q_values(state))
```

### PPOAgent

Proximal Policy Optimization with actor-critic.

```python
from src.agents import PPOAgent

agent = PPOAgent(
    state_dim=4,
    action_dim=2,
    learning_rate=0.0003,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    entropy_coef=0.01,
    n_steps=2048,
    n_epochs=10,
)

# Training loop
action = agent.select_action(state, explore=True)
agent.store_transition(state, action, reward, done)
metrics = agent.update(transition)  # Updates when buffer full
```

### Heuristic Agents

Rule-based baselines for comparison.

```python
from src.agents.heuristic_agents import (
    AlwaysStayAgent,
    BurnoutThresholdAgent,
    ConservativeAgent,
    create_heuristic_agent,
)

# Create specific agent
agent = BurnoutThresholdAgent(state_dim=4, action_dim=2, burnout_threshold=0.6)

# Create by name
agent = create_heuristic_agent("conservative", state_dim=4, action_dim=2)
```

---

## Failure Detection

### CombinedFailureDetector

```python
from src.failure_detection import (
    CombinedFailureDetector,
    RewardHackingDetector,
    PolicyCollapseDetector,
    NonstationarityDetector,
    ValueExplosionDetector,
)

# Create detector
detector = CombinedFailureDetector([
    RewardHackingDetector(),
    PolicyCollapseDetector(entropy_threshold=0.1),
    NonstationarityDetector(),
    ValueExplosionDetector(q_value_threshold=1000),
])

# During training
detector.update({
    "episode": episode,
    "reward": reward,
    "entropy": agent.get_policy_entropy(),
    "mean_q": q_values.mean(),
    "burnout": state[1],
})

# Check for failures
failures = detector.detect_all()
for failure in failures:
    print(failure)

# Get summary
print(detector.get_summary())  # Count by failure type
```

### FailureTaxonomy

```python
from src.failure_detection.taxonomy import FailureTaxonomy, FailureType

# Get all failure info
taxonomy = FailureTaxonomy.get_all()

# Get specific failure info
info = FailureTaxonomy.get_failure_info(FailureType.REWARD_HACKING)
fixes = FailureTaxonomy.get_mitigation(FailureType.POLICY_COLLAPSE)

# Generate documentation
print(FailureTaxonomy.to_markdown_table())
print(FailureTaxonomy.generate_playbook())
```

---

## Introspection Tools

### EntropyTracker

```python
from src.introspection import EntropyTracker

tracker = EntropyTracker(window_size=100)

# During training
for episode in training:
    entropy = agent.get_policy_entropy()
    tracker.update(entropy, episode)

# Analyze
print(tracker.get_trend())  # 'decreasing', 'collapsed', 'stable', etc.
print(tracker.get_diagnostics())

# Visualize
tracker.plot(save_path="entropy.png")
```

### StateHeatmap

```python
from src.introspection import StateHeatmap

heatmap = StateHeatmap(
    bins=(20, 20),
    x_label="Salary",
    y_label="Burnout"
)

# Record states
for state in episode_states:
    heatmap.record(state[:2], action)

# Analyze
print(f"State coverage: {heatmap.get_coverage():.1%}")

# Visualize
heatmap.plot(include_actions=True, action_names=["STAY", "SWITCH"])
```

### CounterfactualAnalyzer

```python
from src.introspection import CounterfactualAnalyzer

analyzer = CounterfactualAnalyzer(env, agent, gamma=0.99)

# Analyze a decision
result = analyzer.analyze(
    state=current_state,
    original_action=0,  # What agent chose
    horizon=10
)

print(result.summary())
print(f"Alternative return: {result.alternative_return:.2f}")
print(f"Was better choice: {result.was_better_choice}")
```

### PhasePlotter

```python
from src.introspection import PhasePlotter

plotter = PhasePlotter(x_label="Salary", y_label="Burnout")

# Record trajectories
for episode in training:
    plotter.add_trajectory(states, actions, rewards, episode)

# Visualize
plotter.plot_single(trajectory, color_by="time")
plotter.plot_multiple(n_trajectories=20, selection="best")
plotter.plot_evolution()  # Show learning progress
```

---

## Experiments

### ExperimentProtocol

```python
from src.experiments import ExperimentProtocol

protocol = ExperimentProtocol(
    name="my_experiment",
    description="Testing reward hacking",
    seed=42,
    
    env_type="career",
    env_params={"burnout_rate": 0.1},
    
    reward_type="short_term_v1",
    reward_params={"lambda_burnout": 0.5},
    
    agent_type="dqn",
    agent_params={"learning_rate": 0.001},
    
    n_episodes=1000,
    eval_interval=50,
    
    hypothesis="Agent will exploit low burnout penalty",
    expected_failure="reward_hacking",
)

# Save/load
protocol.save("experiment.yaml")
protocol = ExperimentProtocol.load("experiment.yaml")
```

### ExperimentRunner

```python
from src.experiments import ExperimentRunner

runner = ExperimentRunner(
    enable_failure_detection=True,
    enable_introspection=True,
    verbose=True
)

result = runner.run(protocol)

print(f"Final reward: {result.final_eval_reward:.2f}")
print(f"Failures detected: {len(result.detected_failures)}")

# Save results
result.save("result.json")
print(result.generate_report())
```

### ExperimentComparator

```python
from src.experiments import ExperimentComparator

comparator = ExperimentComparator()
comparator.add_result("exp1/result.json")
comparator.add_result("exp2/result.json")

# Compare
print(comparator.get_summary_table())
print(comparator.get_best_experiment("final_eval_reward"))

# Visualize
comparator.plot_learning_curves(save_path="comparison.png")
comparator.plot_failure_comparison()

# Generate report
print(comparator.generate_comparison_report())
```

---

## Utilities

### Reproducibility

```python
from src.utils.reproducibility import set_global_seed, get_reproducibility_info

# Set all seeds
set_global_seed(42)

# Log system info
info = get_reproducibility_info()
print(info)  # Python version, PyTorch version, CUDA, etc.
```

### Metrics

```python
from src.utils.metrics import MetricsTracker, compute_returns, compute_advantages

tracker = MetricsTracker(rolling_window=100)
tracker.add("reward", 10.5)
tracker.add_multiple({"loss": 0.1, "entropy": 0.5})

print(tracker.get_rolling_mean("reward"))
print(tracker.get_summary())

# Export
df = tracker.to_dataframe()
```

### Configuration

```python
from src.utils.config import Config, load_config

# Create config
config = Config(name="my_experiment", seed=42)
config.save("config.yaml")

# Load config
config = load_config("config.yaml")
```
