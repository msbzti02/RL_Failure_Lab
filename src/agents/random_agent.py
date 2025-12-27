   
import numpy as np
from typing import Any, Dict
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
           
    name = "random"
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seed: int = None,
        **kwargs
    ):
                   
        super().__init__(state_dim, action_dim, **kwargs)
        
        self.rng = np.random.RandomState(seed)
        self._action_counts = np.zeros(action_dim)
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
                                     
        self._total_steps += 1
        action = self.rng.randint(0, self.action_dim)
        self._action_counts[action] += 1
        return action
    
    def update(self, transition: Dict[str, Any]) -> Dict[str, float]:
                                                 
        return {}
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
                                          
        return np.ones(self.action_dim) / self.action_dim
    
    def get_policy_entropy(self) -> float:
                                                            
        return np.log(self.action_dim)
    
    def get_diagnostics(self) -> Dict[str, Any]:
                                       
        diag = super().get_diagnostics()
        diag["action_distribution"] = self._action_counts / max(1, self._action_counts.sum())
        return diag
