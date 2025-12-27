   
import numpy as np
from typing import Any, Dict
from .base_reward import BaseReward
from .registry import register_reward

@register_reward("sparse_v1")
class SparseReward(BaseReward):
           
    name = "sparse"
    version = "v1"
    description = "Reward only at goal achievement"
    intended_behavior = "Learn to reach goals through exploration"
    known_failure_modes = [
        "exploration_failure",
        "credit_assignment_difficulty",
        "random_walk_to_goal"
    ]
    
    def __init__(
        self,
        salary_threshold: float = 0.7,                          
        burnout_threshold: float = 0.3,                          
        success_bonus: float = 10.0,
        failure_penalty: float = -5.0,
        step_penalty: float = 0.0,                                                  
        **kwargs
    ):
                   
        super().__init__(
            salary_threshold=salary_threshold,
            burnout_threshold=burnout_threshold,
            success_bonus=success_bonus,
            failure_penalty=failure_penalty,
            step_penalty=step_penalty,
            **kwargs
        )
        self.salary_threshold = salary_threshold
        self.burnout_threshold = burnout_threshold
        self.success_bonus = success_bonus
        self.failure_penalty = failure_penalty
        self.step_penalty = step_penalty
    
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
        
        reward = self.step_penalty                             
        
        if terminated:
                                                    
            reward += self.failure_penalty
        elif truncated:
                                                        
            goal_achieved = (
                salary_norm >= self.salary_threshold and
                burnout <= self.burnout_threshold
            )
            if goal_achieved:
                reward += self.success_bonus
        
        self.update(reward)
        return reward

@register_reward("sparse_milestones_v1")
class SparseMilestonesReward(BaseReward):
           
    name = "sparse_milestones"
    version = "v1"
    description = "Sparse reward with salary milestone bonuses"
    intended_behavior = "Learn to progress through salary levels"
    known_failure_modes = [
        "milestone_camping",
        "ignoring_burnout_for_milestones"
    ]
    
    def __init__(
        self,
        milestones: list = [0.3, 0.5, 0.7, 0.9],                                
        milestone_bonus: float = 2.0,
        final_bonus: float = 10.0,
        burnout_penalty: float = -5.0,
        **kwargs
    ):
                   
        super().__init__(
            milestones=milestones,
            milestone_bonus=milestone_bonus,
            final_bonus=final_bonus,
            burnout_penalty=burnout_penalty,
            **kwargs
        )
        self.milestones = sorted(milestones)
        self.milestone_bonus = milestone_bonus
        self.final_bonus = final_bonus
        self.burnout_penalty = burnout_penalty
        
        self._achieved_milestones: set = set()
        self._max_salary: float = 0.0
    
    def reset(self) -> None:
                                       
        super().reset()
        self._achieved_milestones.clear()
        self._max_salary = 0.0
    
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
        
        reward = 0.0
        
        self._max_salary = max(self._max_salary, salary_norm)
        
        for milestone in self.milestones:
            if milestone not in self._achieved_milestones and salary_norm >= milestone:
                self._achieved_milestones.add(milestone)
                reward += self.milestone_bonus
        
        if terminated:
            reward += self.burnout_penalty
        elif truncated:
                                                   
            if burnout <= 0.3 and salary_norm >= 0.6:
                reward += self.final_bonus
        
        self.update(reward)
        return reward
    
    def get_info(self) -> Dict[str, Any]:
                                     
        info = super().get_info()
        info["achieved_milestones"] = len(self._achieved_milestones)
        info["max_salary"] = self._max_salary
        return info
