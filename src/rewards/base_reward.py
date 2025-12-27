   
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np

class BaseReward(ABC):
           
    name: str = "base"
    version: str = "v1"
    description: str = "Abstract base reward"
    intended_behavior: str = ""
    known_failure_modes: list = []
    
    def __init__(self, **params):
                   
        self.params = params
        self._episode_step = 0
        self._cumulative_reward = 0.0
    
    @abstractmethod
    def compute(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        env_info: Dict[str, Any],
        terminated: bool,
        truncated: bool
    ) -> float:
                   
        pass
    
    def reset(self) -> None:
                                                        
        self._episode_step = 0
        self._cumulative_reward = 0.0
    
    def update(self, reward: float) -> None:
                                                                
        self._episode_step += 1
        self._cumulative_reward += reward
    
    def get_info(self) -> Dict[str, Any]:
                                                                      
        return {
            "name": self.name,
            "version": self.version,
            "episode_step": self._episode_step,
            "cumulative_reward": self._cumulative_reward,
            "params": self.params,
        }
    
    @classmethod
    def get_documentation(cls) -> Dict[str, Any]:
                                                              
        return {
            "name": cls.name,
            "version": cls.version,
            "description": cls.description,
            "intended_behavior": cls.intended_behavior,
            "known_failure_modes": cls.known_failure_modes,
        }
    
    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params_str})"
