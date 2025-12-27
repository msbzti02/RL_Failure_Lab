   
from typing import Dict, Any, Optional
from collections import deque
import numpy as np

from .detector import FailureDetector
from .taxonomy import FailureType, FailureReport

class PolicyCollapseDetector(FailureDetector):
           
    failure_type = FailureType.POLICY_COLLAPSE
    
    def __init__(
        self,
        window_size: int = 100,
        entropy_threshold: float = 0.1,
        dominance_threshold: float = 0.95,
        min_samples: int = 100,
        **kwargs
    ):
                   
        super().__init__(window_size)
        self.entropy_threshold = entropy_threshold
        self.dominance_threshold = dominance_threshold
        self.min_samples = min_samples
        
        self._entropies = deque(maxlen=window_size)
        self._action_counts = None
        self._total_actions = 0
    
    def update(self, metrics: Dict[str, Any]) -> None:
                                      
        super().update(metrics)
        
        entropy = metrics.get("entropy", metrics.get("policy_entropy"))
        if entropy is not None:
            self._entropies.append(entropy)
        
        action = metrics.get("action")
        action_dist = metrics.get("action_distribution")
        
        if action is not None:
            if self._action_counts is None:
                                                                       
                n_actions = len(action_dist) if action_dist is not None else 2
                self._action_counts = np.zeros(n_actions)
            
            self._action_counts[action] += 1
            self._total_actions += 1
        
        if action_dist is not None:
                                                      
            probs = np.array(action_dist)
            probs = probs[probs > 0]
            dist_entropy = -np.sum(probs * np.log(probs))
            self._entropies.append(dist_entropy)
    
    def detect(self) -> Optional[FailureReport]:
                                                
        if not self._active:
            return None
        
        if len(self._entropies) >= 10:
            mean_entropy = np.mean(list(self._entropies)[-20:])                  
            
            if mean_entropy < self.entropy_threshold:
                return self._create_report(
                    severity="high",
                    signal_values={
                        "mean_entropy": mean_entropy,
                        "min_entropy": np.min(self._entropies),
                        "entropy_threshold": self.entropy_threshold,
                    },
                    description=(
                        f"Policy entropy ({mean_entropy:.4f}) has dropped below "
                        f"threshold ({self.entropy_threshold}). Policy has collapsed."
                    ),
                    recommended_fix=(
                        "Increase entropy coefficient in PPO, or slow down epsilon "
                        "decay in DQN. Consider adding exploration bonuses."
                    ),
                )
        
        if self._total_actions >= self.min_samples:
            action_probs = self._action_counts / self._total_actions
            max_prob = np.max(action_probs)
            dominant_action = np.argmax(action_probs)
            
            if max_prob > self.dominance_threshold:
                return self._create_report(
                    severity="high" if max_prob > 0.99 else "medium",
                    signal_values={
                        "dominant_action": int(dominant_action),
                        "dominant_action_prob": max_prob,
                        "action_distribution": action_probs.tolist(),
                    },
                    description=(
                        f"Action {dominant_action} taken in {max_prob:.1%} of cases. "
                        f"Policy has collapsed to a single action."
                    ),
                    recommended_fix=(
                        "Force more exploration through higher epsilon/temperature "
                        "or add entropy regularization to the policy loss."
                    ),
                )
        
        return None
    
    def reset(self) -> None:
                                   
        super().reset()
        self._entropies.clear()
        self._action_counts = None
        self._total_actions = 0
