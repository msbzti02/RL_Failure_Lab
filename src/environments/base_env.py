   
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class BaseEnv(gym.Env, ABC):
           
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode
        self._current_step = 0
        self._episode_count = 0
                                                         
        self._np_random, _ = gym.utils.seeding.np_random(None)
    
    @abstractmethod
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
                   
        self._np_random, _ = gym.utils.seeding.np_random(seed)
        self._current_step = 0
        self._episode_count += 1
    
    @abstractmethod
    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
                   
        self._current_step += 1
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
                   
        pass
    
    def render(self, mode: Optional[str] = None) -> Optional[str]:
                   
        mode = mode or self.render_mode
        if mode == "ansi":
            return self._render_ansi()
        elif mode == "human":
            print(self._render_ansi())
        return None
    
    def _render_ansi(self) -> str:
                                                                     
        return f"Step {self._current_step}"
    
    @property
    def current_step(self) -> int:
                                                     
        return self._current_step
    
    @property
    def episode_count(self) -> int:
                                           
        return self._episode_count
    
    def get_state_dict(self) -> Dict[str, Any]:
                   
        return {
            "step": self._current_step,
            "episode": self._episode_count,
        }
    
    def seed(self, seed: Optional[int] = None) -> list:
                                                                
        self._np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
