   
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Any, Dict, List, Optional
from pathlib import Path
from collections import deque

from .base_agent import BaseAgent

class ActorCritic(nn.Module):
                                                
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64]
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers) if layers else nn.Identity()
        
        self.actor = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1], 1)
        )
    
    def forward(self, x: torch.Tensor) -> tuple:
                                                          
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value.squeeze(-1)
    
    def get_action(self, x: torch.Tensor) -> tuple:
                                                            
        action_probs, value = self.forward(x)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action), value, dist.entropy()

class RolloutBuffer:
                                          
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def get(self) -> Dict[str, np.ndarray]:
        return {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "values": np.array(self.values),
            "log_probs": np.array(self.log_probs),
            "dones": np.array(self.dones),
        }
    
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def __len__(self) -> int:
        return len(self.states)

class PPOAgent(BaseAgent):
           
    name = "ppo"
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        n_steps: int = 2048,
        batch_size: int = 64,
        hidden_dims: list = [64, 64],
        device: str = "auto",
        **kwargs
    ):
                   
        super().__init__(state_dim, action_dim, device, **kwargs)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.batch_size = batch_size
        
        self.actor_critic = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        self.buffer = RolloutBuffer()
        
        self._recent_entropies = deque(maxlen=100)
        self._recent_kl_divs = deque(maxlen=100)
        self._recent_value_losses = deque(maxlen=100)
        self._action_counts = np.zeros(action_dim)
        
        self._last_value = 0.0
        self._last_log_prob = 0.0
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
                                                 
        self._total_steps += 1
        
        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            
            if explore:
                action, log_prob, value, entropy = self.actor_critic.get_action(state_tensor)
                action = action.item()
                self._last_log_prob = log_prob.item()
                self._last_value = value.item()
                self._recent_entropies.append(entropy.item())
            else:
                               
                action_probs, value = self.actor_critic(state_tensor)
                action = action_probs.argmax().item()
                self._last_value = value.item()
                self._last_log_prob = np.log(action_probs[0, action].item() + 1e-8)
        
        self._action_counts[action] += 1
        return action
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        done: bool
    ):
                                                       
        self.buffer.push(
            state,
            action,
            reward,
            self._last_value,
            self._last_log_prob,
            done
        )
    
    def update(self, transition: Dict[str, Any]) -> Dict[str, float]:
                   
        self.store_transition(
            transition["state"],
            transition["action"],
            transition["reward"],
            transition["done"]
        )
        
        if len(self.buffer) < self.n_steps:
            return {}
        
        with torch.no_grad():
            final_state = self._to_tensor(transition["next_state"]).unsqueeze(0)
            _, final_value = self.actor_critic(final_state)
            final_value = 0.0 if transition["done"] else final_value.item()
        
        data = self.buffer.get()
        advantages, returns = self._compute_gae(
            data["rewards"],
            data["values"],
            data["dones"],
            final_value
        )
        
        states = self._to_tensor(data["states"])
        actions = self._to_tensor(data["actions"], dtype=torch.long)
        old_log_probs = self._to_tensor(data["log_probs"])
        advantages = self._to_tensor(advantages)
        returns = self._to_tensor(returns)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        update_count = 0
        
        indices = np.arange(len(states))
        
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                action_probs, values = self.actor_critic(batch_states)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(values, batch_returns)
                
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                
                with torch.no_grad():
                    kl = (batch_old_log_probs - new_log_probs).mean().item()
                    total_kl += abs(kl)
                
                update_count += 1
        
        self._updates += 1
        self.buffer.clear()
        
        avg_kl = total_kl / update_count
        self._recent_kl_divs.append(avg_kl)
        self._recent_value_losses.append(total_value_loss / update_count)
        
        return {
            "loss": total_loss / update_count,
            "policy_loss": total_policy_loss / update_count,
            "value_loss": total_value_loss / update_count,
            "entropy": total_entropy / update_count,
            "kl_divergence": avg_kl,
        }
    
    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        final_value: float
    ) -> tuple:
                                                       
        advantages = np.zeros_like(rewards)
        last_advantage = 0
        
        values_extended = np.append(values, final_value)
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_extended[t + 1] * (1 - dones[t]) - values_extended[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        return advantages, returns
    
    def get_value(self, state: np.ndarray) -> float:
                                             
        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            _, value = self.actor_critic(state_tensor)
            return value.item()
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
                                                  
        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            action_probs, _ = self.actor_critic(state_tensor)
            return self._to_numpy(action_probs.squeeze())
    
    def get_policy_entropy(self) -> float:
                                                   
        if self._recent_entropies:
            return np.mean(self._recent_entropies)
        return np.log(self.action_dim)
    
    def get_diagnostics(self) -> Dict[str, Any]:
                                            
        diag = super().get_diagnostics()
        
        diag["buffer_size"] = len(self.buffer)
        diag["action_distribution"] = self._action_counts / max(1, self._action_counts.sum())
        
        if self._recent_entropies:
            diag["mean_entropy"] = np.mean(self._recent_entropies)
            diag["min_entropy"] = np.min(self._recent_entropies)
            
            if diag["min_entropy"] < 0.1:
                diag["entropy_collapse_warning"] = True
        
        if self._recent_kl_divs:
            diag["mean_kl"] = np.mean(self._recent_kl_divs)
        
        if self._recent_value_losses:
            diag["mean_value_loss"] = np.mean(self._recent_value_losses)
        
        return diag
    
    def save(self, path: str) -> None:
                               
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self._total_steps,
            "episodes": self._episodes,
            "updates": self._updates,
            "action_counts": self._action_counts,
        }, path)
    
    def load(self, path: str) -> None:
                               
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self._total_steps = checkpoint["total_steps"]
        self._episodes = checkpoint["episodes"]
        self._updates = checkpoint["updates"]
        self._action_counts = checkpoint["action_counts"]
