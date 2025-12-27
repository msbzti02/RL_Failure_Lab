# RL Failure Lab - Experiment Results

**Total experiments:** 45
**Agents tested:** 9
**Reward functions tested:** 5

## Full Results Table

| Rank | Agent | Reward | Eval Mean | Train Mean | Failures | Entropy Trend |
|------|-------|--------|-----------|------------|----------|---------------|
| 1 | ppo | short_term_v2 | 179.08 | 158.73 | 71 | decreasing |
| 2 | ppo | short_term_v1 | 178.70 | 155.17 | 92 | collapsed |
| 3 | always_switch | long_term_shaped_v1 | 178.04 | 174.47 | 162 | always_low |
| 4 | always_switch | short_term_v1 | 177.28 | 175.19 | 162 | always_low |
| 5 | dqn | short_term_v2 | 176.55 | 173.93 | 119 | always_low |
| 6 | dqn | risk_sensitive_v1 | 176.43 | 174.47 | 126 | always_low |
| 7 | dqn | short_term_v1 | 176.22 | 172.99 | 125 | always_low |
| 8 | ppo | long_term_shaped_v1 | 176.04 | 155.40 | 71 | decreasing |
| 9 | ppo | risk_sensitive_v1 | 175.37 | 154.51 | 71 | decreasing |
| 10 | dqn | long_term_shaped_v1 | 174.50 | 173.36 | 123 | always_low |
| 11 | always_switch | risk_sensitive_v1 | 173.49 | 174.88 | 162 | always_low |
| 12 | always_switch | short_term_v2 | 172.87 | 175.97 | 162 | always_low |
| 13 | aggressive | short_term_v2 | 123.07 | 122.78 | 162 | always_low |
| 14 | aggressive | short_term_v1 | 107.41 | 107.98 | 162 | always_low |
| 15 | aggressive | risk_sensitive_v1 | 101.01 | 101.74 | 162 | always_low |
| 16 | aggressive | long_term_shaped_v1 | 93.31 | 106.79 | 162 | always_low |
| 17 | adaptive | long_term_shaped_v1 | 51.84 | 48.64 | 223 | always_low |
| 18 | adaptive | short_term_v1 | 50.09 | 51.93 | 233 | always_low |
| 19 | adaptive | risk_sensitive_v1 | 41.58 | 47.88 | 233 | always_low |
| 20 | adaptive | short_term_v2 | 40.08 | 40.99 | 211 | always_low |
| 21 | burnout_threshold | long_term_shaped_v1 | 32.88 | 28.55 | 233 | always_low |
| 22 | burnout_threshold | short_term_v1 | 28.35 | 29.64 | 233 | always_low |
| 23 | burnout_threshold | risk_sensitive_v1 | 27.13 | 27.99 | 233 | always_low |
| 24 | dqn | sparse_v1 | 10.00 | 8.60 | 71 | stable |
| 25 | ppo | sparse_v1 | 10.00 | 8.35 | 102 | collapsed |
| 26 | always_switch | sparse_v1 | 10.00 | 9.70 | 162 | always_low |
| 27 | random | short_term_v2 | 8.23 | 7.36 | 193 | stable |
| 28 | random | risk_sensitive_v1 | 3.56 | 7.65 | 193 | stable |
| 29 | aggressive | sparse_v1 | 3.00 | 4.60 | 162 | always_low |
| 30 | random | short_term_v1 | 2.34 | 9.16 | 193 | stable |
| 31 | adaptive | sparse_v1 | 1.00 | 0.40 | 153 | always_low |
| 32 | burnout_threshold | sparse_v1 | 0.00 | 0.00 | 91 | always_low |
| 33 | random | long_term_shaped_v1 | -0.05 | 4.62 | 193 | stable |
| 34 | burnout_threshold | short_term_v2 | -1.13 | -2.32 | 185 | always_low |
| 35 | random | sparse_v1 | -5.00 | -5.00 | 51 | stable |
| 36 | always_stay | sparse_v1 | -5.00 | -5.00 | 91 | always_low |
| 37 | conservative | sparse_v1 | -5.00 | -5.00 | 91 | always_low |
| 38 | always_stay | short_term_v1 | -6.88 | -7.13 | 91 | always_low |
| 39 | always_stay | risk_sensitive_v1 | -8.31 | -8.30 | 91 | always_low |
| 40 | conservative | short_term_v1 | -10.18 | -10.22 | 162 | always_low |
| 41 | conservative | risk_sensitive_v1 | -10.40 | -11.50 | 162 | always_low |
| 42 | always_stay | long_term_shaped_v1 | -11.01 | -11.07 | 91 | always_low |
| 43 | always_stay | short_term_v2 | -11.04 | -11.15 | 91 | always_low |
| 44 | conservative | long_term_shaped_v1 | -14.64 | -13.94 | 162 | always_low |
| 45 | conservative | short_term_v2 | -14.73 | -13.74 | 162 | always_low |

## Top 5 Best Performers

**1. ppo + short_term_v2**
   - Eval reward: 179.08
   - Train reward: 158.73
   - Failures: 71

**2. ppo + short_term_v1**
   - Eval reward: 178.70
   - Train reward: 155.17
   - Failures: 92

**3. always_switch + long_term_shaped_v1**
   - Eval reward: 178.04
   - Train reward: 174.47
   - Failures: 162

**4. always_switch + short_term_v1**
   - Eval reward: 177.28
   - Train reward: 175.19
   - Failures: 162

**5. dqn + short_term_v2**
   - Eval reward: 176.55
   - Train reward: 173.93
   - Failures: 119

## Bottom 5 Performers

**41. conservative + risk_sensitive_v1**
   - Eval reward: -10.40
   - Train reward: -11.50
   - Failures: 162

**42. always_stay + long_term_shaped_v1**
   - Eval reward: -11.01
   - Train reward: -11.07
   - Failures: 91

**43. always_stay + short_term_v2**
   - Eval reward: -11.04
   - Train reward: -11.15
   - Failures: 91

**44. conservative + long_term_shaped_v1**
   - Eval reward: -14.64
   - Train reward: -13.94
   - Failures: 162

**45. conservative + short_term_v2**
   - Eval reward: -14.73
   - Train reward: -13.74
   - Failures: 162

## Failure Analysis

| Failure Type | Experiments Affected |
|--------------|---------------------|
| policy_collapse | 41 |
| nonstationarity | 37 |
| reward_hacking | 13 |

## Average Performance by Agent

| Agent | Avg Eval Reward |
|-------|-----------------|
| ppo | 143.84 |
| dqn | 142.74 |
| always_switch | 142.34 |
| aggressive | 85.56 |
| adaptive | 36.92 |
| burnout_threshold | 17.45 |
| random | 1.81 |
| always_stay | -8.45 |
| conservative | -10.99 |

## Average Performance by Reward Function

| Reward | Avg Eval Reward |
|--------|-----------------|
| short_term_v1 | 78.15 |
| long_term_shaped_v1 | 75.66 |
| risk_sensitive_v1 | 75.54 |
| short_term_v2 | 74.78 |
| sparse_v1 | 2.11 |