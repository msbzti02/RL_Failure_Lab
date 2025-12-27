# RL Stress Test Report

Generated: 2025-12-27 21:05

## Summary

| Test | Name | Result | Key Finding |
|------|------|--------|-------------|
| T0.1 | Deterministic | PASS | {'diff': 0.0}... |
| T0.2 | Random Baseline | PASS | {'mean_reward': 10.555392490984813}... |
| T1.1 | Salary Trap | PASS | {'switch_ratio': 0.8542063189950514, 'mean_burnout... |
| T2.2 | Sparse Reward | PASS | {'dense_eval': 174.7541407471464, 'sparse_eval': 9... |
| T3.1 | High Scale | PASS | {'max_q': 1065.2559814453125}... |
| T3.3 | Collapse | PASS | {'max_action': 0.9958}... |
| T4.1 | Regime Shift | PASS | {'stable': 175.23268402354296, 'recession': 0.1646... |
| T7.1 | Heuristic | PASS | {'comparison': {'dqn': 176.20233658282547, 'ppo': ... |
| T8.1 | Seeds | PASS | {'mean': 175.30979234892757, 'std': 1.575324411373... |

## Key Lessons

1. **Reward Hacking**: Single-objective rewards are pathological
2. **Sparse Rewards**: Credit assignment is hard
3. **Scale Matters**: Reward scaling breaks value methods
4. **Exploration**: Fast decay = collapse
5. **Generalization**: Train != test distribution
