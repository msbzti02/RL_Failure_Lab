   
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import torch

class BaseAgent(ABC):
           
    name: str = "base"
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = "auto",
        **kwargs
    ):
                   
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self._total_steps = 0
        self._episodes = 0
        self._updates = 0
    
    @abstractmethod
    def select_action(
        self, 
        state: np.ndarray, 
        explore: bool = True
    ) -> int:
                   
        pass
    
    @abstractmethod
    def update(
        self, 
        transition: Dict[str, Any]
    ) -> Dict[str, float]:
                   
        pass
    
    def get_policy_entropy(self) -> float:
                   
        return 0.0                                       
    
    def get_action_distribution(self, state: np.ndarray) -> np.ndarray:
                   
        return np.ones(self.action_dim) / self.action_dim
    
    def get_value(self, state: np.ndarray) -> float:
                   
        return 0.0           
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
                   
        return np.zeros(self.action_dim)           
    
    def on_episode_end(self) -> None:
                                                                      
        self._episodes += 1
    
    def get_diagnostics(self) -> Dict[str, Any]:
                   
        return {
            "name": self.name,
            "total_steps": self._total_steps,
            "episodes": self._episodes,
            "updates": self._updates,
            "policy_entropy": self.get_policy_entropy(),
        }
    
    def save(self, path: str) -> None:
                   
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "total_steps": self._total_steps,
            "episodes": self._episodes,
            "updates": self._updates,
        }, path)
    
    def load(self, path: str) -> None:
                   
        checkpoint = torch.load(path, map_location=self.device)
        self._total_steps = checkpoint.get("total_steps", 0)
        self._episodes = checkpoint.get("episodes", 0)
        self._updates = checkpoint.get("updates", 0)
    
    def _to_tensor(self, x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
                                                              
        return torch.tensor(x, dtype=dtype, device=self.device)
    
    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
                                            
        return x.detach().cpu().numpy()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(state_dim={self.state_dim}, action_dim={self.action_dim})"
