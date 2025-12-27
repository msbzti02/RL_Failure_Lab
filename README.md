# RL Failure Lab 🧪

> **A modular reinforcement learning experimentation framework for studying failure modes in sequential decision-making.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 What Is This?

**This is not an RL demo.** It's a controlled experimental framework for:

- **Studying failure modes** in RL systems (reward hacking, policy collapse, instability)
- **Comparing algorithms** fairly across environments and reward functions
- **Diagnosing problems** systematically with automatic failure detection
- **Learning RL deeply** by understanding what breaks and why

Perfect for:
- 🎓 **Students** learning RL beyond the basics
- 🔬 **Researchers** studying RL robustness and safety
- 💼 **Engineers** building production RL systems

---

## 🏗️ Architecture

```
rl-failure-lab/
├── src/
│   ├── environments/        # Parameterized environments
│   │   ├── career_env.py    # Career simulation (salary, burnout, switching)
│   │   └── param_engine.py  # Dynamic environment configuration
│   │
│   ├── rewards/             # Versioned reward function registry
│   │   ├── short_term.py    # Immediate optimization
│   │   ├── long_term_shaped.py  # Potential-based shaping
│   │   ├── risk_sensitive.py    # Variance/CVaR penalties
│   │   ├── sparse.py        # Goal-only rewards
│   │   └── delayed.py       # End-of-episode rewards
│   │
│   ├── agents/              # RL and baseline agents
│   │   ├── dqn_agent.py     # Deep Q-Network
│   │   ├── ppo_agent.py     # Proximal Policy Optimization
│   │   └── heuristic_agents.py  # Rule-based baselines
│   │
│   ├── failure_detection/   # Automatic failure detection
│   │   ├── reward_hacking.py
│   │   ├── policy_collapse.py
│   │   ├── nonstationarity.py
│   │   └── value_explosion.py
│   │
│   ├── introspection/       # Policy analysis tools
│   │   ├── entropy_tracker.py
│   │   ├── state_heatmap.py
│   │   ├── counterfactual.py
│   │   └── phase_plots.py
│   │
│   └── experiments/         # Experiment infrastructure
│       ├── protocol.py      # Reproducible specifications
│       ├── runner.py        # Full instrumented execution
│       └── comparator.py    # Cross-experiment analysis
│
├── scripts/
│   ├── demo.py              # Quick start demo
│   ├── run_all_experiments.py   # Full benchmark suite
│   └── stress_tests_simple.py   # Failure mode tests
│
└── docs/
    ├── api_reference.md     # Complete API documentation
    └── failure_playbook.md  # Debugging guide
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rl-failure-lab.git
cd rl-failure-lab

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Run the Demo

```bash
python scripts/demo.py
```

### Run Your First Experiment

```python
from src.experiments import ExperimentRunner, ExperimentProtocol

protocol = ExperimentProtocol(
    name="my_first_experiment",
    agent_type="dqn",
    reward_type="short_term_v1",
    n_episodes=500,
)

runner = ExperimentRunner()
result = runner.run(protocol)

print(f"Final reward: {result.final_eval_reward:.2f}")
print(f"Failures detected: {len(result.detected_failures)}")
```

---

## 📊 Key Features

### 1. Environment Parameterization Engine

Test agent robustness across different conditions:

```python
from src.environments import CareerEnv
from src.environments.param_engine import EnvironmentParams

# Stress test with recession
params = EnvironmentParams.for_regime("recession")
env = CareerEnv(params)

# Or customize parameters
params = EnvironmentParams(
    burnout_rate=0.2,          # High burnout accumulation
    salary_volatility=0.3,     # Volatile market
    switching_risk=0.5,        # Risky job switches
)
```

**Available regimes:** `stable`, `boom`, `recession`, `volatile`

### 2. Versioned Reward Function Registry

10 reward functions with documented failure modes:

| Reward | Behavior | Known Failure Modes |
|--------|----------|---------------------|
| `short_term_v1` | `salary - λ*burnout` | Reward hacking, myopic |
| `short_term_v2` | Quadratic burnout penalty | Over-penalization |
| `long_term_shaped_v1` | Potential-based shaping | Potential mismatch |
| `risk_sensitive_v1` | Variance penalty | Over-conservative |
| `sparse_v1` | Goal-only reward | Exploration failure |
| `delayed_v1` | Episode-end reward | Credit assignment |

```python
from src.rewards import get_reward, list_rewards

# List all available rewards
print(list_rewards())

# Get a specific reward function
reward_fn = get_reward("short_term_v1", lambda_burnout=1.5)
```

### 3. Automatic Failure Detection

Real-time detection of common RL failures:

```python
from src.failure_detection import (
    CombinedFailureDetector,
    RewardHackingDetector,
    PolicyCollapseDetector,
    ValueExplosionDetector,
)

detector = CombinedFailureDetector([
    RewardHackingDetector(),
    PolicyCollapseDetector(entropy_threshold=0.1),
    ValueExplosionDetector(q_value_threshold=1000),
])

# During training
detector.update(metrics)
failures = detector.detect_all()

for failure in failures:
    print(f"[{failure.severity}] {failure.failure_type}: {failure.description}")
```

### 4. Policy Introspection Tools

Understand what your agent believes:

```python
from src.introspection import EntropyTracker, StateHeatmap, PhasePlotter

# Track exploration over time
entropy_tracker = EntropyTracker()
entropy_tracker.update(agent.get_policy_entropy(), episode)
entropy_tracker.plot(save_path="entropy.png")

# Visualize state visitation
heatmap = StateHeatmap()
for state in episode_states:
    heatmap.record(state[:2], action)
heatmap.plot(include_actions=True, action_names=["STAY", "SWITCH"])

# Plot trajectories in phase space
plotter = PhasePlotter()
plotter.add_trajectory(states, actions, rewards, episode)
plotter.plot_multiple(n_trajectories=20, selection="best")
```

### 5. Human Baselines

Compare RL to simple rules:

```python
from src.agents.heuristic_agents import (
    AlwaysStayAgent,
    BurnoutThresholdAgent,
    ConservativeAgent,
    AdaptiveAgent,
)

# Create a rule-based agent
heuristic = BurnoutThresholdAgent(
    state_dim=4, 
    action_dim=2, 
    burnout_threshold=0.6
)

# Compare to RL
# If RL < heuristic: explain why honestly
```

---

## 🧪 Experiment Results

### Full Benchmark (9 agents × 5 rewards = 45 experiments)

**Top 5 Performers:**

| Rank | Agent | Reward | Score |
|------|-------|--------|-------|
| 1 | **PPO** | short_term_v2 | 179.08 |
| 2 | PPO | short_term_v1 | 178.70 |
| 3 | always_switch | long_term_shaped_v1 | 178.04 |
| 4 | DQN | short_term_v2 | 176.55 |
| 5 | DQN | risk_sensitive_v1 | 176.43 |

**Average by Agent:**
- PPO: 143.84
- DQN: 142.74
- always_switch: 142.34
- random: 1.81
- conservative: -10.99

### Stress Test Results

All 9 tests passed by exhibiting expected failure behaviors:

| Test | Expected Failure | Observed |
|------|------------------|----------|
| T1.1 Salary Trap | Burnout explosion | 85% switches, burnout spike |
| T2.2 Sparse Reward | Credit assignment failure | Dense 175 >> Sparse 9 |
| T3.1 High Scale | Q-value explosion | max_q = 1065 |
| T3.3 Exploration Collapse | Single action dominance | 99.58% one action |
| T4.1 Regime Shift | Performance drop | 175 → 0.16 |

---

## 📚 Documentation

- **[API Reference](docs/api_reference.md)** — Complete API documentation
- **[Failure Playbook](docs/failure_playbook.md)** — Diagnosis and fixes

### Quick Debugging Reference

| If you see... | It usually means... | Try... |
|---------------|---------------------|--------|
| Action entropy → 0 | Policy collapse | ↑ entropy coefficient |
| High reward + high burnout | Reward hacking | Redesign reward |
| Oscillating curves | Non-stationarity | ↓ learning rate |
| NaN/Inf values | Value explosion | Gradient clipping |

---

## 🎓 What You'll Learn

Using this framework, you'll understand:

1. **Reward Hacking** — How agents exploit reward loopholes
2. **Policy Collapse** — Why exploration matters
3. **Credit Assignment** — The challenge of delayed rewards
4. **Distribution Shift** — Why train ≠ test
5. **Baseline Comparison** — When complex is worse than simple
6. **Reproducibility** — Why seeds matter

---

## 🛠️ Running Experiments

### Full Comparison Suite

```bash
python scripts/run_all_experiments.py
```

Runs 45 experiments (9 agents × 5 rewards), generates:
- `experiments/RESULTS_REPORT.md` — Summary report
- `experiments/results/full_comparison_*.json` — Raw data

### Stress Tests

```bash
python scripts/stress_tests_simple.py
```

Runs failure mode tests:
- T0: Sanity checks
- T1: Reward hacking
- T2: Credit assignment
- T3: Instability
- T4: Non-stationarity
- T7: Baseline comparison
- T8: Reproducibility

---

## 📁 Project Files

```
├── requirements.txt         # Dependencies
├── setup.py                 # Package installation
├── README.md                # This file
│
├── src/                     # Source code
│   ├── environments/        # 3 files
│   ├── rewards/             # 7 files
│   ├── agents/              # 5 files
│   ├── failure_detection/   # 6 files
│   ├── introspection/       # 5 files
│   ├── experiments/         # 5 files
│   └── utils/               # 4 files
│
├── scripts/                 # Runnable scripts
│   ├── demo.py
│   ├── run_all_experiments.py
│   ├── stress_tests_simple.py
│   ├── generate_report.py
│   └── show_results.py
│
├── docs/                    # Documentation
│   ├── api_reference.md
│   └── failure_playbook.md
│
└── experiments/             # Results (generated)
    ├── RESULTS_REPORT.md
    └── STRESS_TEST_REPORT.md
```

---

## 📋 Requirements

- Python 3.10+
- PyTorch 2.0+
- Gymnasium
- NumPy, Pandas, Matplotlib, Seaborn
- TensorBoard (optional, for logging)
- tqdm, PyYAML

Install all:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- [ ] Additional environments (GridWorld, CartPole variants)
- [ ] More failure detectors (gradient pathologies, representation collapse)
- [ ] Visualization dashboard (Streamlit/Gradio)
- [ ] Additional agents (SAC, A2C, model-based)
- [ ] POMDP tests (partial observability)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

This project was inspired by:
- The OpenAI Gym/Gymnasium project
- Spinning Up in Deep RL
- "Deep Reinforcement Learning that Matters" (Henderson et al.)
- The RL debugging community

---

## 📬 Contact

Questions or feedback? Open an issue or reach out!

---

<p align="center">
  <b>Built for learning. Designed for breaking.</b><br>
  <i>Because understanding failure is the path to success.</i>
</p>
