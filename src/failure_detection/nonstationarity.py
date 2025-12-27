   
from typing import Dict, Any, Optional
from collections import deque
import numpy as np

from .detector import FailureDetector
from .taxonomy import FailureType, FailureReport

class NonstationarityDetector(FailureDetector):
           
    failure_type = FailureType.NONSTATIONARITY
    
    def __init__(
        self,
        window_size: int = 100,
        variance_threshold: float = 2.0,
        oscillation_threshold: float = 3,
        min_samples: int = 50,
        **kwargs
    ):
                   
        super().__init__(window_size)
        self.variance_threshold = variance_threshold
        self.oscillation_threshold = oscillation_threshold
        self.min_samples = min_samples
        
        self._rewards = deque(maxlen=window_size)
        self._reward_trends = deque(maxlen=window_size // 10)
    
    def update(self, metrics: Dict[str, Any]) -> None:
                                      
        super().update(metrics)
        
        reward = metrics.get("reward", metrics.get("episode_reward", 0))
        self._rewards.append(reward)
        
        if len(self._rewards) >= 10:
            recent_avg = np.mean(list(self._rewards)[-10:])
            older_avg = np.mean(list(self._rewards)[-20:-10]) if len(self._rewards) >= 20 else recent_avg
            
            self._reward_trends.append(1 if recent_avg > older_avg else -1)
    
    def detect(self) -> Optional[FailureReport]:
                                                 
        if not self._active or len(self._rewards) < self.min_samples:
            return None
        
        reward_variance = np.var(self._rewards)
        reward_mean = np.mean(self._rewards)
        cv = np.sqrt(reward_variance) / (abs(reward_mean) + 1e-8)                            
        
        if reward_variance > self.variance_threshold:
                                           
            if len(self._reward_trends) >= 5:
                                         
                trends = list(self._reward_trends)
                changes = sum(1 for i in range(1, len(trends)) if trends[i] != trends[i-1])
                
                if changes >= self.oscillation_threshold:
                    return self._create_report(
                        severity="medium",
                        signal_values={
                            "reward_variance": reward_variance,
                            "reward_cv": cv,
                            "direction_changes": changes,
                            "reward_mean": reward_mean,
                        },
                        description=(
                            f"Training shows oscillating pattern with {changes} direction "
                            f"changes. Reward variance is {reward_variance:.2f}. "
                            f"This indicates non-stationary dynamics."
                        ),
                        recommended_fix=(
                            "Consider using curriculum learning, reducing learning rate, "
                            "or adding temporal regularization to stabilize training."
                        ),
                    )
            
            return self._create_report(
                severity="low",
                signal_values={
                    "reward_variance": reward_variance,
                    "reward_cv": cv,
                    "reward_mean": reward_mean,
                },
                description=(
                    f"High reward variance ({reward_variance:.2f}) detected. "
                    f"Coefficient of variation: {cv:.2f}. Training may be unstable."
                ),
                recommended_fix=(
                    "Monitor for oscillation. Consider normalizing rewards or "
                    "reducing learning rate."
                ),
            )
        
        return None
    
    def reset(self) -> None:
                                   
        super().reset()
        self._rewards.clear()
        self._reward_trends.clear()
