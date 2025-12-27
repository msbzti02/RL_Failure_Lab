   
import numpy as np
from typing import Any, Dict
from .base_reward import BaseReward
from .registry import register_reward

@register_reward("short_term_v1")
class ShortTermReward(BaseReward):
           
    name = "short_term"
    version = "v1"
    description = "Immediate reward based on current salary and burnout"
    intended_behavior = "Maximize salary while managing burnout in the short term"
    known_failure_modes = [
        "reward_hacking",
        "myopic_behavior",
        "burnout_spiral"
    ]
    
    def __init__(
        self,
        lambda_burnout: float = 1.0,
        salary_weight: float = 1.0,
        normalize_salary: bool = True,
        min_salary: float = 30000.0,
        max_salary: float = 300000.0,
        **kwargs
    ):
                   
        super().__init__(
            lambda_burnout=lambda_burnout,
            salary_weight=salary_weight,
            normalize_salary=normalize_salary,
            min_salary=min_salary,
            max_salary=max_salary,
            **kwargs
        )
        self.lambda_burnout = lambda_burnout
        self.salary_weight = salary_weight
        self.normalize_salary = normalize_salary
        self.min_salary = min_salary
        self.max_salary = max_salary
    
    def compute(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        env_info: Dict[str, Any],
        terminated: bool,
        truncated: bool
    ) -> float:
                                        
        salary = env_info.get("salary", next_state[0] * (self.max_salary - self.min_salary) + self.min_salary)
        burnout = env_info.get("burnout", next_state[1])
        
        if self.normalize_salary:
            salary_norm = (salary - self.min_salary) / (self.max_salary - self.min_salary)
        else:
            salary_norm = salary / 100000                       
        
        reward = self.salary_weight * salary_norm - self.lambda_burnout * burnout
        
        if terminated and burnout >= 1.0:
            reward -= 1.0
        
        self.update(reward)
        return reward

@register_reward("short_term_v2")
class ShortTermRewardV2(BaseReward):
           
    name = "short_term"
    version = "v2"
    description = "Short-term reward with non-linear burnout penalty"
    intended_behavior = "Strongly penalize high burnout states"
    known_failure_modes = [
        "over_conservative",
        "salary_avoidance"
    ]
    
    def __init__(
        self,
        lambda_burnout: float = 2.0,
        salary_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(lambda_burnout=lambda_burnout, salary_weight=salary_weight, **kwargs)
        self.lambda_burnout = lambda_burnout
        self.salary_weight = salary_weight
    
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
        
        reward = self.salary_weight * salary_norm - self.lambda_burnout * (burnout ** 2)
        
        if terminated:
            reward -= 2.0
        
        self.update(reward)
        return reward
