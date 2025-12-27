   
import numpy as np
from typing import Any, Dict, Optional
from .base_reward import BaseReward
from .registry import register_reward

@register_reward("long_term_shaped_v1")
class LongTermShapedReward(BaseReward):
           
    name = "long_term_shaped"
    version = "v1"
    description = "Potential-based shaping for long-term planning"
    intended_behavior = "Encourage states with high future potential"
    known_failure_modes = [
        "potential_function_mismatch",
        "shaping_reward_hacking"
    ]
    
    def __init__(
        self,
        gamma: float = 0.99,
        salary_potential_weight: float = 1.0,
        burnout_potential_weight: float = 2.0,
        tenure_potential_weight: float = 0.5,
        base_lambda_burnout: float = 1.0,
        **kwargs
    ):
                   
        super().__init__(
            gamma=gamma,
            salary_potential_weight=salary_potential_weight,
            burnout_potential_weight=burnout_potential_weight,
            tenure_potential_weight=tenure_potential_weight,
            base_lambda_burnout=base_lambda_burnout,
            **kwargs
        )
        self.gamma = gamma
        self.salary_potential_weight = salary_potential_weight
        self.burnout_potential_weight = burnout_potential_weight
        self.tenure_potential_weight = tenure_potential_weight
        self.base_lambda_burnout = base_lambda_burnout
        
        self._prev_potential: Optional[float] = None
    
    def reset(self) -> None:
                                       
        super().reset()
        self._prev_potential = None
    
    def _compute_potential(self, state: np.ndarray) -> float:
                   
        salary_norm = state[0]
        burnout = state[1]
        tenure_norm = state[2]
        
        salary_potential = self.salary_potential_weight * salary_norm
        burnout_potential = -self.burnout_potential_weight * (burnout ** 2)
        tenure_potential = self.tenure_potential_weight * min(tenure_norm, 0.5)
        
        return salary_potential + burnout_potential + tenure_potential
    
    def compute(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        env_info: Dict[str, Any],
        terminated: bool,
        truncated: bool
    ) -> float:
                                    
        salary_norm = next_state[0]
        burnout = next_state[1]
        base_reward = salary_norm - self.base_lambda_burnout * burnout
        
        current_potential = self._compute_potential(next_state)
        
        if self._prev_potential is None:
                                    
            shaping_reward = 0.0
        else:
                                                               
            shaping_reward = self.gamma * current_potential - self._prev_potential
        
        if not (terminated or truncated):
            self._prev_potential = current_potential
        else:
                                            
            shaping_reward = -self._prev_potential if self._prev_potential else 0.0
            self._prev_potential = None
        
        reward = base_reward + shaping_reward
        
        if terminated and burnout >= 1.0:
            reward -= 5.0
        
        self.update(reward)
        return reward
    
    def get_info(self) -> Dict[str, Any]:
                                        
        info = super().get_info()
        info["current_potential"] = self._prev_potential
        return info

@register_reward("long_term_shaped_v2")
class LongTermShapedRewardV2(LongTermShapedReward):
           
    name = "long_term_shaped"
    version = "v2"
    description = "Shaped reward with trajectory momentum"
    intended_behavior = "Reward consistent improvement over time"
    known_failure_modes = [
        "momentum_exploitation",
        "oscillation_for_improvement"
    ]
    
    def __init__(self, momentum_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.momentum_weight = momentum_weight
        self._prev_salary = None
        self._prev_burnout = None
    
    def reset(self) -> None:
        super().reset()
        self._prev_salary = None
        self._prev_burnout = None
    
    def compute(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        env_info: Dict[str, Any],
        terminated: bool,
        truncated: bool
    ) -> float:
                                                  
        base_shaped_reward = super().compute(
            state, action, next_state, env_info, terminated, truncated
        )
                                            
        self._episode_step -= 1
        self._cumulative_reward -= base_shaped_reward
        
        salary = next_state[0]
        burnout = next_state[1]
        
        momentum_bonus = 0.0
        if self._prev_salary is not None and self._prev_burnout is not None:
            salary_improvement = salary - self._prev_salary
            burnout_improvement = self._prev_burnout - burnout                           
            
            momentum_bonus = self.momentum_weight * (salary_improvement + burnout_improvement)
        
        self._prev_salary = salary
        self._prev_burnout = burnout
        
        reward = base_shaped_reward + momentum_bonus
        self.update(reward)
        return reward
