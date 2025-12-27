   
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field

@dataclass
class RollingStats:
                                                     
    window_size: int = 100
    _values: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        self._values = deque(maxlen=self.window_size)
    
    def add(self, value: float) -> None:
        self._values.append(value)
    
    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return np.mean(self._values)
    
    @property
    def std(self) -> float:
        if len(self._values) < 2:
            return 0.0
        return np.std(self._values)
    
    @property
    def min(self) -> float:
        if not self._values:
            return 0.0
        return np.min(self._values)
    
    @property
    def max(self) -> float:
        if not self._values:
            return 0.0
        return np.max(self._values)
    
    @property
    def count(self) -> int:
        return len(self._values)
    
    def get_stats(self) -> dict:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "count": self.count,
        }

class MetricsTracker:
           
    def __init__(self, rolling_window: int = 100):
        self.rolling_window = rolling_window
        self._history: Dict[str, List[float]] = {}
        self._rolling: Dict[str, RollingStats] = {}
        self._step = 0
    
    def add(self, name: str, value: float, step: Optional[int] = None) -> None:
                                        
        if name not in self._history:
            self._history[name] = []
            self._rolling[name] = RollingStats(window_size=self.rolling_window)
        
        self._history[name].append(value)
        self._rolling[name].add(value)
        
        if step is not None:
            self._step = step
    
    def add_multiple(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
                                           
        for name, value in metrics.items():
            self.add(name, value, step=None)
        if step is not None:
            self._step = step
    
    def get_history(self, name: str) -> List[float]:
                                           
        return self._history.get(name, [])
    
    def get_last(self, name: str, default: float = 0.0) -> float:
                                                      
        history = self._history.get(name, [])
        return history[-1] if history else default
    
    def get_rolling_mean(self, name: str) -> float:
                                           
        if name in self._rolling:
            return self._rolling[name].mean
        return 0.0
    
    def get_rolling_std(self, name: str) -> float:
                                                         
        if name in self._rolling:
            return self._rolling[name].std
        return 0.0
    
    def get_rolling_stats(self, name: str) -> dict:
                                                 
        if name in self._rolling:
            return self._rolling[name].get_stats()
        return {}
    
    def get_all_means(self) -> Dict[str, float]:
                                                
        return {name: stats.mean for name, stats in self._rolling.items()}
    
    def get_summary(self) -> Dict[str, dict]:
                                                     
        return {name: stats.get_stats() for name, stats in self._rolling.items()}
    
    def reset(self) -> None:
                                        
        self._history.clear()
        self._rolling.clear()
        self._step = 0
    
    def to_dataframe(self):
                                                         
        import pandas as pd
        
        max_len = max(len(v) for v in self._history.values()) if self._history else 0
        
        data = {}
        for name, values in self._history.items():
            padded = values + [np.nan] * (max_len - len(values))
            data[name] = padded
        
        return pd.DataFrame(data)

def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
                                                  
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

def compute_advantages(
    rewards: List[float],
    values: List[float],
    next_value: float,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> tuple:
           
    advantages = []
    gae = 0
    
    values_extended = values + [next_value]
    
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values_extended[t + 1] - values_extended[t]
        gae = delta + gamma * gae_lambda * gae
        advantages.insert(0, gae)
    
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
           
    var_y = np.var(y_true)
    if var_y == 0:
        return 0.0
    return 1 - np.var(y_true - y_pred) / var_y

if __name__ == "__main__":
          
    tracker = MetricsTracker(rolling_window=10)
    
    for i in range(50):
        reward = np.random.normal(10, 2)
        loss = np.random.exponential(0.1)
        tracker.add_multiple({"reward": reward, "loss": loss})
    
    print("Summary:")
    for name, stats in tracker.get_summary().items():
        print(f"  {name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
