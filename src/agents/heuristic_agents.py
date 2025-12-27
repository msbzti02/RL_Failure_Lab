   
import numpy as np
from typing import Any, Dict
from .base_agent import BaseAgent

class HeuristicAgent(BaseAgent):
                                           
    name = "heuristic"
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, **kwargs)
        self._action_counts = np.zeros(action_dim)
    
    def update(self, transition: Dict[str, Any]) -> Dict[str, float]:
                                               
        return {}
    
    def get_diagnostics(self) -> Dict[str, Any]:
        diag = super().get_diagnostics()
        diag["action_distribution"] = self._action_counts / max(1, self._action_counts.sum())
        return diag

class AlwaysStayAgent(HeuristicAgent):
           
    name = "always_stay"
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        self._total_steps += 1
        action = 0        
        self._action_counts[action] += 1
        return action
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
        return np.array([1.0, 0.0])
    
    def get_policy_entropy(self) -> float:
        return 0.0                 

class AlwaysSwitchAgent(HeuristicAgent):
           
    name = "always_switch"
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        self._total_steps += 1
        action = 1          
        self._action_counts[action] += 1
        return action
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
        return np.array([0.0, 1.0])
    
    def get_policy_entropy(self) -> float:
        return 0.0

class BurnoutThresholdAgent(HeuristicAgent):
           
    name = "burnout_threshold"
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        burnout_threshold: float = 0.6,
        **kwargs
    ):
                   
        super().__init__(state_dim, action_dim, **kwargs)
        self.burnout_threshold = burnout_threshold
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        self._total_steps += 1
        
        burnout = state[1]
        
        if burnout > self.burnout_threshold:
            action = 1          
        else:
            action = 0        
        
        self._action_counts[action] += 1
        return action
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
        burnout = state[1]
        if burnout > self.burnout_threshold:
            return np.array([0.0, 1.0])
        return np.array([1.0, 0.0])

class ConservativeAgent(HeuristicAgent):
           
    name = "conservative"
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        burnout_threshold: float = 0.5,
        market_threshold: float = 0.0,
        **kwargs
    ):
                   
        super().__init__(state_dim, action_dim, **kwargs)
        self.burnout_threshold = burnout_threshold
        self.market_threshold = market_threshold
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        self._total_steps += 1
        
        burnout = state[1]
        market = state[3]
        
        if burnout > self.burnout_threshold and market > self.market_threshold:
            action = 1          
        else:
            action = 0        
        
        self._action_counts[action] += 1
        return action

class AggressiveAgent(HeuristicAgent):
           
    name = "aggressive"
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        burnout_threshold: float = 0.3,
        market_threshold: float = 0.2,
        tenure_threshold: int = 20,                                
        **kwargs
    ):
                   
        super().__init__(state_dim, action_dim, **kwargs)
        self.burnout_threshold = burnout_threshold
        self.market_threshold = market_threshold
        self.tenure_threshold = tenure_threshold
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        self._total_steps += 1
        
        burnout = state[1]
        tenure_norm = state[2]
        market = state[3]
        
        should_switch = (
            market > self.market_threshold or               
            burnout > self.burnout_threshold or                    
            tenure_norm > self.tenure_threshold / 100                      
        )
        
        action = 1 if should_switch else 0
        self._action_counts[action] += 1
        return action

class AdaptiveAgent(HeuristicAgent):
           
    name = "adaptive"
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        recession_burnout_threshold: float = 0.7,
        boom_burnout_threshold: float = 0.3,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, **kwargs)
        self.recession_burnout_threshold = recession_burnout_threshold
        self.boom_burnout_threshold = boom_burnout_threshold
    
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        self._total_steps += 1
        
        burnout = state[1]
        market = state[3]
        
        if market < -0.3:
                                             
            threshold = self.recession_burnout_threshold
        elif market > 0.3:
                                 
            threshold = self.boom_burnout_threshold
        else:
                                   
            threshold = 0.5
        
        action = 1 if burnout > threshold else 0
        self._action_counts[action] += 1
        return action

HEURISTIC_AGENTS = {
    "always_stay": AlwaysStayAgent,
    "always_switch": AlwaysSwitchAgent,
    "burnout_threshold": BurnoutThresholdAgent,
    "conservative": ConservativeAgent,
    "aggressive": AggressiveAgent,
    "adaptive": AdaptiveAgent,
}

def create_heuristic_agent(name: str, state_dim: int, action_dim: int, **kwargs):
                                           
    if name not in HEURISTIC_AGENTS:
        raise ValueError(f"Unknown heuristic agent: {name}. Available: {list(HEURISTIC_AGENTS.keys())}")
    return HEURISTIC_AGENTS[name](state_dim, action_dim, **kwargs)

if __name__ == "__main__":
          
    print("Available heuristic agents:")
    for name, cls in HEURISTIC_AGENTS.items():
        agent = cls(4, 2)
        print(f"  {name}: {agent.__class__.__doc__.strip().split(chr(10))[0]}")
