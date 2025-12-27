   
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

from .base_agent import BaseAgent

class QNetwork(nn.Module):
                            
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [64, 64]
    ):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
                                   
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent(BaseAgent):
           
    name = "dqn"
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        hidden_dims: list = [64, 64],
        target_update_freq: int = 100,
        tau: float = 0.005,                           
        device: str = "auto",
        **kwargs
    ):
                   
        super().__init__(state_dim, action_dim, device, **kwargs)
        
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        
        self.q_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self._action_counts = np.zeros(action_dim)
        self._recent_q_values = deque(maxlen=1000)
        self._recent_losses = deque(maxlen=100)
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
                                                            
        self._total_steps += 1
        
        if explore and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = self._to_tensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax().item()
                
                self._recent_q_values.append(q_values.cpu().numpy().flatten())
        
        self._action_counts[action] += 1
        return action
    
    def update(self, transition: Dict[str, Any]) -> Dict[str, float]:
                                                           
        self.replay_buffer.push(
            transition["state"],
            transition["action"],
            transition["reward"],
            transition["next_state"],
            transition["done"]
        )
        
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        states = self._to_tensor(states)
        actions = self._to_tensor(actions, dtype=torch.long)
        rewards = self._to_tensor(rewards)
        next_states = self._to_tensor(next_states)
        dones = self._to_tensor(dones)
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        
        self.optimizer.step()
        
        self._updates += 1
        self._recent_losses.append(loss.item())
        
        if self._updates % self.target_update_freq == 0:
            self._soft_update_target()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "mean_q": current_q.mean().item(),
            "max_q": current_q.max().item(),
        }
    
    def _soft_update_target(self):
                                            
        for target_param, param in zip(
            self.target_network.parameters(), 
            self.q_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
                                              
        with torch.no_grad():
            state_tensor = self._to_tensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return self._to_numpy(q_values.squeeze())
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
                                                                         
        q_values = self.get_q_values(state)
                                    
        exp_q = np.exp(q_values - q_values.max())
        return exp_q / exp_q.sum()
    
    def get_policy_entropy(self) -> float:
                                                               
        total = self._action_counts.sum()
        if total == 0:
            return np.log(self.action_dim)                   
        
        probs = self._action_counts / total
        probs = probs[probs > 0]                
        return -np.sum(probs * np.log(probs))
    
    def get_diagnostics(self) -> Dict[str, Any]:
                                            
        diag = super().get_diagnostics()
        
        diag["epsilon"] = self.epsilon
        diag["buffer_size"] = len(self.replay_buffer)
        diag["action_distribution"] = self._action_counts / max(1, self._action_counts.sum())
        
        if self._recent_losses:
            diag["mean_loss"] = np.mean(self._recent_losses)
        
        if self._recent_q_values:
            q_array = np.array(list(self._recent_q_values))
            diag["mean_q"] = np.mean(q_array)
            diag["max_q"] = np.max(q_array)
            diag["q_std"] = np.std(q_array)
            
            if np.any(np.isnan(q_array)) or np.any(np.abs(q_array) > 1000):
                diag["value_explosion_warning"] = True
        
        return diag
    
    def save(self, path: str) -> None:
                               
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self._total_steps,
            "episodes": self._episodes,
            "updates": self._updates,
            "action_counts": self._action_counts,
        }, path)
    
    def load(self, path: str) -> None:
                               
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self._total_steps = checkpoint["total_steps"]
        self._episodes = checkpoint["episodes"]
        self._updates = checkpoint["updates"]
        self._action_counts = checkpoint["action_counts"]
