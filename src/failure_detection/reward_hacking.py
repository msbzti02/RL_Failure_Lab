   
from typing import Dict, Any, Optional
from collections import deque
import numpy as np

from .detector import FailureDetector
from .taxonomy import FailureType, FailureReport

class RewardHackingDetector(FailureDetector):
           
    failure_type = FailureType.REWARD_HACKING
    
    def __init__(
        self,
        window_size: int = 100,
        reward_burnout_threshold: float = 0.3,
        min_correlation_samples: int = 50,
        **kwargs
    ):
                   
        super().__init__(window_size)
        self.reward_burnout_threshold = reward_burnout_threshold
        self.min_correlation_samples = min_correlation_samples
        
        self._rewards = deque(maxlen=window_size)
        self._burnouts = deque(maxlen=window_size)
        self._outcomes = deque(maxlen=window_size)                         
        self._high_reward_high_burnout_count = 0
    
    def update(self, metrics: Dict[str, Any]) -> None:
                                              
        super().update(metrics)
        
        reward = metrics.get("reward", metrics.get("episode_reward", 0))
        burnout = metrics.get("burnout", metrics.get("final_burnout", 0))
        
        self._rewards.append(reward)
        self._burnouts.append(burnout)
        
        if reward > 0.5 and burnout > 0.6:
            self._high_reward_high_burnout_count += 1
        
        outcome = 1.0 if burnout < 0.5 and reward > 0 else 0.0
        self._outcomes.append(outcome)
    
    def detect(self) -> Optional[FailureReport]:
                                               
        if not self._active or len(self._rewards) < self.min_correlation_samples:
            return None
        
        suspicious_ratio = self._high_reward_high_burnout_count / len(self._rewards)
        if suspicious_ratio > self.reward_burnout_threshold:
            return self._create_report(
                severity="high",
                signal_values={
                    "suspicious_episode_ratio": suspicious_ratio,
                    "mean_reward": np.mean(self._rewards),
                    "mean_burnout": np.mean(self._burnouts),
                },
                description=(
                    f"Agent achieving high rewards despite high burnout in "
                    f"{suspicious_ratio:.1%} of episodes. This suggests reward hacking."
                ),
                recommended_fix=(
                    "Redesign reward function to more strongly penalize burnout, "
                    "or add explicit constraints on burnout levels."
                ),
            )
        
        if len(self._rewards) >= self.min_correlation_samples:
            correlation = np.corrcoef(list(self._rewards), list(self._outcomes))[0, 1]
            
            if correlation < -0.3:                                           
                return self._create_report(
                    severity="critical",
                    signal_values={
                        "reward_outcome_correlation": correlation,
                        "mean_reward": np.mean(self._rewards),
                        "mean_outcome_quality": np.mean(self._outcomes),
                    },
                    description=(
                        f"Negative correlation ({correlation:.2f}) between reward and "
                        f"outcome quality indicates severe reward hacking."
                    ),
                    recommended_fix=(
                        "Reward function is fundamentally misaligned. "
                        "Consider complete redesign or using human feedback."
                    ),
                )
        
        return None
    
    def reset(self) -> None:
                                   
        super().reset()
        self._rewards.clear()
        self._burnouts.clear()
        self._outcomes.clear()
        self._high_reward_high_burnout_count = 0
