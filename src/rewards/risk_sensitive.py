   
import numpy as np
from typing import Any, Dict, List
from collections import deque
from .base_reward import BaseReward
from .registry import register_reward

@register_reward("risk_sensitive_v1")
class RiskSensitiveReward(BaseReward):
           
    name = "risk_sensitive"
    version = "v1"
    description = "Variance-penalized reward for risk aversion"
    intended_behavior = "Prefer stable, predictable outcomes"
    known_failure_modes = [
        "over_conservative",
        "stuck_in_safe_states",
        "exploration_avoidance"
    ]
    
    def __init__(
        self,
        alpha: float = 0.5,
        window_size: int = 20,
        lambda_burnout: float = 1.0,
        min_variance_samples: int = 5,
        **kwargs
    ):
                   
        super().__init__(
            alpha=alpha,
            window_size=window_size,
            lambda_burnout=lambda_burnout,
            min_variance_samples=min_variance_samples,
            **kwargs
        )
        self.alpha = alpha
        self.window_size = window_size
        self.lambda_burnout = lambda_burnout
        self.min_variance_samples = min_variance_samples
        
        self._reward_history: deque = deque(maxlen=window_size)
    
    def reset(self) -> None:
                                   
        super().reset()
        self._reward_history.clear()
    
    def _compute_base_reward(self, state: np.ndarray, env_info: Dict[str, Any]) -> float:
                                                          
        salary_norm = state[0]
        burnout = state[1]
        return salary_norm - self.lambda_burnout * burnout
    
    def compute(
        self,
        state: np.ndarray,
        action: int,
        next_state: np.ndarray,
        env_info: Dict[str, Any],
        terminated: bool,
        truncated: bool
    ) -> float:
                                            
        base_reward = self._compute_base_reward(next_state, env_info)
        
        self._reward_history.append(base_reward)
        
        variance_penalty = 0.0
        if len(self._reward_history) >= self.min_variance_samples:
            variance = np.var(list(self._reward_history))
            variance_penalty = self.alpha * variance
        
        reward = base_reward - variance_penalty
        
        if terminated:
            reward -= 2.0
        
        self.update(reward)
        return reward
    
    def get_info(self) -> Dict[str, Any]:
                                       
        info = super().get_info()
        if len(self._reward_history) >= 2:
            info["reward_variance"] = np.var(list(self._reward_history))
            info["reward_std"] = np.std(list(self._reward_history))
        return info

@register_reward("risk_sensitive_cvar_v1")
class CVaRReward(BaseReward):
           
    name = "risk_sensitive_cvar"
    version = "v1"
    description = "CVaR-based risk-sensitive reward"
    intended_behavior = "Avoid worst-case outcomes, not just variance"
    known_failure_modes = [
        "extreme_conservatism",
        "paralysis_in_risky_states"
    ]
    
    def __init__(
        self,
        cvar_alpha: float = 0.2,                          
        cvar_weight: float = 0.5,
        window_size: int = 50,
        lambda_burnout: float = 1.0,
        **kwargs
    ):
                   
        super().__init__(
            cvar_alpha=cvar_alpha,
            cvar_weight=cvar_weight,
            window_size=window_size,
            lambda_burnout=lambda_burnout,
            **kwargs
        )
        self.cvar_alpha = cvar_alpha
        self.cvar_weight = cvar_weight
        self.window_size = window_size
        self.lambda_burnout = lambda_burnout
        
        self._outcome_history: deque = deque(maxlen=window_size)
    
    def reset(self) -> None:
                                    
        super().reset()
        self._outcome_history.clear()
    
    def _compute_cvar(self) -> float:
                                        
        if len(self._outcome_history) < 10:
            return 0.0
        
        outcomes = np.array(self._outcome_history)
        n_worst = max(1, int(len(outcomes) * self.cvar_alpha))
        
        sorted_outcomes = np.sort(outcomes)
        worst_outcomes = sorted_outcomes[:n_worst]
        
        return np.mean(worst_outcomes)
    
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
        
        base_reward = salary_norm - self.lambda_burnout * burnout
        
        self._outcome_history.append(base_reward)
        
        cvar = self._compute_cvar()
        cvar_penalty = self.cvar_weight * max(0, -cvar)                                 
        
        reward = base_reward - cvar_penalty
        
        if terminated:
            reward -= 3.0
        
        self.update(reward)
        return reward
    
    def get_info(self) -> Dict[str, Any]:
                                   
        info = super().get_info()
        info["cvar"] = self._compute_cvar()
        return info
