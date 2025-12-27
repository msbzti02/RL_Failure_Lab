   
import numpy as np
from typing import Any, Dict, List
from .base_reward import BaseReward
from .registry import register_reward

@register_reward("delayed_v1")
class DelayedReward(BaseReward):
           
    name = "delayed"
    version = "v1"
    description = "Reward provided only at episode termination"
    intended_behavior = "Learn long-horizon credit assignment"
    known_failure_modes = [
        "credit_assignment_failure",
        "slow_learning",
        "high_variance_updates"
    ]
    
    def __init__(
        self,
        salary_weight: float = 1.0,
        burnout_weight: float = 2.0,
        survival_bonus: float = 5.0,
        burnout_penalty: float = -10.0,
        normalize_by_steps: bool = True,
        **kwargs
    ):
                   
        super().__init__(
            salary_weight=salary_weight,
            burnout_weight=burnout_weight,
            survival_bonus=survival_bonus,
            burnout_penalty=burnout_penalty,
            normalize_by_steps=normalize_by_steps,
            **kwargs
        )
        self.salary_weight = salary_weight
        self.burnout_weight = burnout_weight
        self.survival_bonus = survival_bonus
        self.burnout_penalty = burnout_penalty
        self.normalize_by_steps = normalize_by_steps
        
        self._salaries: List[float] = []
        self._burnouts: List[float] = []
    
    def reset(self) -> None:
                                        
        super().reset()
        self._salaries.clear()
        self._burnouts.clear()
    
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
        
        self._salaries.append(salary_norm)
        self._burnouts.append(burnout)
        
        if not (terminated or truncated):
            self.update(0.0)
            return 0.0
        
        reward = self._compute_final_reward(terminated)
        
        self.update(reward)
        return reward
    
    def _compute_final_reward(self, terminated_by_burnout: bool) -> float:
                                                      
        if not self._salaries:
            return 0.0
        
        final_salary = self._salaries[-1]
        max_salary = max(self._salaries)
        avg_salary = np.mean(self._salaries)
        total_burnout_cost = sum(self._burnouts)
        final_burnout = self._burnouts[-1]
        
        salary_score = self.salary_weight * (0.3 * final_salary + 0.3 * max_salary + 0.4 * avg_salary)
        
        if self.normalize_by_steps:
            burnout_penalty = self.burnout_weight * (total_burnout_cost / len(self._burnouts))
        else:
            burnout_penalty = self.burnout_weight * final_burnout
        
        reward = salary_score - burnout_penalty
        
        if terminated_by_burnout:
            reward += self.burnout_penalty
        else:
                                  
            reward += self.survival_bonus
            
            if final_burnout < 0.3 and final_salary > 0.6:
                reward += 3.0
        
        return reward
    
    def get_info(self) -> Dict[str, Any]:
                                               
        info = super().get_info()
        if self._salaries:
            info["trajectory_length"] = len(self._salaries)
            info["max_salary_achieved"] = max(self._salaries)
            info["avg_burnout"] = np.mean(self._burnouts)
        return info

@register_reward("delayed_hindsight_v1") 
class DelayedHindsightReward(BaseReward):
           
    name = "delayed_hindsight"
    version = "v1"
    description = "Delayed reward with hindsight trajectory analysis"
    intended_behavior = "Learn from counterfactual analysis"
    known_failure_modes = [
        "hindsight_bias",
        "overestimation_of_alternatives"
    ]
    
    def __init__(
        self,
        counterfactual_weight: float = 0.3,
        **kwargs
    ):
                   
        super().__init__(counterfactual_weight=counterfactual_weight, **kwargs)
        self.counterfactual_weight = counterfactual_weight
        
        self._trajectory: List[Dict[str, Any]] = []
    
    def reset(self) -> None:
                               
        super().reset()
        self._trajectory.clear()
    
    def compute(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        env_info: Dict[str, Any],
        terminated: bool,
        truncated: bool
    ) -> float:
                                                        
        self._trajectory.append({
            "state": state.copy(),
            "action": action,
            "next_state": next_state.copy(),
            "info": env_info.copy()
        })
        
        if not (terminated or truncated):
            self.update(0.0)
            return 0.0
        
        final_salary = next_state[0]
        final_burnout = next_state[1]
        
        base_reward = final_salary - 2 * final_burnout
        
        n_switches = sum(1 for t in self._trajectory if t["action"] == 1)
        n_stays = len(self._trajectory) - n_switches
        
        if final_burnout > 0.7:
                                                            
            hindsight_bonus = 0.1 * n_stays                             
        elif n_switches > len(self._trajectory) * 0.3:
                                                              
            hindsight_bonus = -0.1 * n_switches
        else:
            hindsight_bonus = 0
        
        reward = base_reward + self.counterfactual_weight * hindsight_bonus
        
        if terminated:
            reward -= 5.0
        else:
            reward += 3.0
        
        self.update(reward)
        return reward
