# RL Failure Diagnosis Playbook

> A practical guide to diagnosing and fixing common RL training failures.

## Quick Reference

| If you see... | It usually means... | Try... |
|---------------|---------------------|--------|
| Action entropy → 0 | Policy collapse | Increase entropy coefficient, slow ε decay |
| High reward + high burnout | Reward hacking | Redesign reward function |
| Oscillating training curves | Non-stationarity | Reduce learning rate, curriculum learning |
| NaN/Inf in metrics | Value explosion | Gradient clipping, learning rate reduction |
| Random-like behavior persists | Exploration failure | Increase exploration, intrinsic motivation |
| Good train, poor eval | Overfitting | Domain randomization, regularization |

---

## Detailed Diagnosis

### Reward Hacking

**Symptom:** High reward achieved but outcomes are poor

**What it looks like:**
- Agent gets high rewards but final state is bad (high burnout)
- Reward curve looks good but qualitative behavior is strange
- Agent finds "loopholes" in the reward function

**Root cause:** Reward function doesn't capture true objectives

**Detection signals:**
```
- Burnout spikes despite positive rewards
- Negative correlation between reward and outcome quality
- Suspicious episode patterns (high reward + high burnout)
```

**How to fix:**
1. **Redesign reward function** - Add explicit penalties for bad states
2. **Use reward shaping** - Add potential-based terms
3. **Increase burnout penalty** - Make consequences more severe
4. **Add constraints** - Hard limits on acceptable states

**Example fix:**
```python
# Before (hackable)
reward = salary_norm - 0.5 * burnout

# After (harder to hack)
reward = salary_norm - 2.0 * (burnout ** 2)
if burnout > 0.8:
    reward -= 5.0  # Hard penalty
```

---

### Policy Collapse

**Symptom:** Agent always takes the same action

**What it looks like:**
- Action distribution: [0.99, 0.01] or [0.01, 0.99]
- Policy entropy drops to near 0
- Agent stops exploring alternatives

**Root cause:** Exploration decays too fast

**Detection signals:**
```
- Action entropy < 0.1
- Single action in >90% of steps
- Q-values for all actions become similar
```

**How to fix:**
1. **Slow down epsilon decay**
   ```python
   # Before
   epsilon_decay = 0.99  # Too fast
   
   # After
   epsilon_decay = 0.9995  # Much slower
   ```

2. **Add entropy bonus (PPO)**
   ```python
   entropy_coef = 0.05  # Increase from 0.01
   ```

3. **Use count-based exploration**
   ```python
   intrinsic_reward = 1.0 / np.sqrt(state_visit_count + 1)
   ```

4. **Lower epsilon end value**
   ```python
   epsilon_end = 0.1  # Keep some exploration
   ```

---

### Non-stationarity

**Symptom:** Oscillating training curves or unstable performance

**What it looks like:**
- Reward goes up, then down, then up again
- Performance never stabilizes
- Different random seeds give wildly different results

**Root cause:** Environment dynamics change due to agent's actions

**Detection signals:**
```
- High reward variance
- Direction changes in moving average
- Unstable value estimates
```

**How to fix:**
1. **Reduce learning rate**
   ```python
   learning_rate = 0.0001  # Slower learning
   ```

2. **Use curriculum learning**
   ```python
   # Start with easier environment
   for difficulty in [0.1, 0.2, 0.3, 0.5, 1.0]:
       env.set_difficulty(difficulty)
       train(agent, env, episodes=200)
   ```

3. **Add target network smoothing**
   ```python
   tau = 0.001  # Slower target updates
   ```

4. **Normalize rewards**
   ```python
   reward = (reward - reward_mean) / (reward_std + 1e-8)
   ```

---

### Value Explosion

**Symptom:** NaN values, divergent training, extreme Q-values

**What it looks like:**
- Q-values: [-1000, 5000] (should be [-10, 10])
- Loss becomes NaN or Inf
- Training crashes

**Root cause:** Poor scaling or unstable learning dynamics

**Detection signals:**
```
- Q-value norms > 100
- Loss > 1000
- Any NaN in metrics
```

**How to fix:**
1. **Apply gradient clipping**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
   ```

2. **Reduce learning rate significantly**
   ```python
   learning_rate = 0.00001  # 10x smaller
   ```

3. **Normalize rewards**
   ```python
   # Clip rewards to reasonable range
   reward = np.clip(reward, -10, 10)
   ```

4. **Use target network with slow updates**
   ```python
   target_update_freq = 1000  # Update less frequently
   tau = 0.001  # Very slow soft updates
   ```

5. **Clip Q-value targets**
   ```python
   target_q = torch.clamp(target_q, -100, 100)
   ```

---

### Exploration Failure

**Symptom:** Agent converges to suboptimal policy early

**What it looks like:**
- Agent finds a "good enough" strategy and stops improving
- Many states never visited
- Performance plateaus below optimal

**Root cause:** Insufficient exploration of state-action space

**Detection signals:**
```
- Low state space coverage
- Quick convergence to local minimum
- Poor performance on novel states
```

**How to fix:**
1. **Increase initial exploration**
   ```python
   epsilon_start = 1.0
   epsilon_end = 0.1  # Keep more exploration
   ```

2. **Add intrinsic motivation**
   ```python
   # Curiosity-driven exploration
   curiosity_reward = prediction_error
   total_reward = extrinsic_reward + 0.1 * curiosity_reward
   ```

3. **Use softmax action selection**
   ```python
   # Temperature-based exploration
   action_probs = softmax(q_values / temperature)
   action = np.random.choice(actions, p=action_probs)
   ```

4. **Domain randomization**
   ```python
   # Vary environment parameters
   env_params = sample_random_params()
   env = create_env(**env_params)
   ```

---

## Diagnostic Checklist

When training goes wrong, check these in order:

### 1. Sanity Checks
- [ ] Is the environment working correctly? Run random agent.
- [ ] Are rewards reasonable? Print and inspect.
- [ ] Is the agent learning anything? Check loss decreasing.

### 2. Exploration
- [ ] What's the current epsilon/entropy?
- [ ] Is action distribution balanced?
- [ ] Are all states being visited?

### 3. Stability
- [ ] Are Q-values in reasonable range?
- [ ] Is loss stable?
- [ ] Are gradients reasonable?

### 4. Reward Design
- [ ] Does high reward correlate with good outcomes?
- [ ] Are there exploitable loopholes?
- [ ] Is the reward dense enough for learning?

### 5. Hyperparameters
- [ ] Is learning rate appropriate?
- [ ] Is batch size sufficient?
- [ ] Is network capacity adequate?

---

## Common Hyperparameter Fixes

| Problem | Hyperparameter | Adjustment |
|---------|---------------|------------|
| Unstable training | Learning rate | Decrease by 10x |
| Slow learning | Learning rate | Increase by 2x |
| Policy collapse | Epsilon decay | Make slower |
| Value explosion | Gradient clipping | Enable/tighten |
| Poor exploration | Entropy coef | Increase |
| High variance | Batch size | Increase |
| Overfitting | Regularization | Add/increase |

---

## Emergency Fixes

When nothing else works:

1. **Reset and try different seed** - Sometimes it's just bad luck
2. **Simplify the environment** - Start with easier version
3. **Check for bugs** - Print everything, inspect manually
4. **Use baseline agent** - Does a simple heuristic work?
5. **Try different algorithm** - Switch DQN ↔ PPO
