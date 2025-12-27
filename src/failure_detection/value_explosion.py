   
from typing import Dict, Any, Optional
from collections import deque
import numpy as np

from .detector import FailureDetector
from .taxonomy import FailureType, FailureReport

class ValueExplosionDetector(FailureDetector):
           
    failure_type = FailureType.VALUE_EXPLOSION
    
    def __init__(
        self,
        window_size: int = 100,
        q_value_threshold: float = 1000.0,
        growth_rate_threshold: float = 2.0,
        min_samples: int = 20,
        **kwargs
    ):
                   
        super().__init__(window_size)
        self.q_value_threshold = q_value_threshold
        self.growth_rate_threshold = growth_rate_threshold
        self.min_samples = min_samples
        
        self._q_values = deque(maxlen=window_size)
        self._losses = deque(maxlen=window_size)
        self._gradient_norms = deque(maxlen=window_size)
        self._nan_detected = False
    
    def update(self, metrics: Dict[str, Any]) -> None:
                                      
        super().update(metrics)
        
        q_value = metrics.get("mean_q", metrics.get("q_value"))
        max_q = metrics.get("max_q")
        
        if q_value is not None:
            if np.isnan(q_value) or np.isinf(q_value):
                self._nan_detected = True
            else:
                self._q_values.append(q_value)
        
        if max_q is not None and not np.isnan(max_q):
            self._q_values.append(max_q)
        
        loss = metrics.get("loss")
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                self._nan_detected = True
            else:
                self._losses.append(loss)
        
        grad_norm = metrics.get("gradient_norm", metrics.get("grad_norm"))
        if grad_norm is not None:
            if np.isnan(grad_norm) or np.isinf(grad_norm):
                self._nan_detected = True
            else:
                self._gradient_norms.append(grad_norm)
    
    def detect(self) -> Optional[FailureReport]:
                                                
        if not self._active:
            return None
        
        if self._nan_detected:
            return self._create_report(
                severity="critical",
                signal_values={
                    "nan_detected": True,
                },
                description=(
                    "NaN or Inf detected in training metrics. "
                    "Training has completely diverged."
                ),
                recommended_fix=(
                    "Apply gradient clipping, reduce learning rate significantly, "
                    "normalize rewards, and check for division by zero in code."
                ),
            )
        
        if len(self._q_values) >= self.min_samples:
            max_q = np.max(np.abs(list(self._q_values)))
            
            if max_q > self.q_value_threshold:
                return self._create_report(
                    severity="high",
                    signal_values={
                        "max_q_value": max_q,
                        "mean_q_value": np.mean(list(self._q_values)[-20:]),
                        "threshold": self.q_value_threshold,
                    },
                    description=(
                        f"Q-values ({max_q:.1f}) have exceeded threshold "
                        f"({self.q_value_threshold}). Values are exploding."
                    ),
                    recommended_fix=(
                        "Apply target network with slower updates, use gradient "
                        "clipping, reduce learning rate, or normalize rewards."
                    ),
                )
            
            if len(self._q_values) >= self.min_samples:
                early_q = np.mean(list(self._q_values)[:self.min_samples//2])
                late_q = np.mean(list(self._q_values)[-self.min_samples//2:])
                
                if early_q != 0:
                    growth_rate = abs(late_q - early_q) / (abs(early_q) + 1e-8)
                    
                    if growth_rate > self.growth_rate_threshold:
                        return self._create_report(
                            severity="medium",
                            signal_values={
                                "growth_rate": growth_rate,
                                "early_q": early_q,
                                "late_q": late_q,
                            },
                            description=(
                                f"Q-value growth rate ({growth_rate:.2f}x) exceeds "
                                f"threshold. Values may be heading toward explosion."
                            ),
                            recommended_fix=(
                                "Monitor closely and consider reducing learning rate "
                                "or adding value clipping."
                            ),
                        )
        
        if len(self._losses) >= 10:
            mean_loss = np.mean(list(self._losses)[-10:])
            
            if mean_loss > 1000:
                return self._create_report(
                    severity="high",
                    signal_values={
                        "mean_loss": mean_loss,
                        "max_loss": np.max(list(self._losses)[-10:]),
                    },
                    description=(
                        f"Loss ({mean_loss:.1f}) has grown very large. "
                        f"Training is unstable."
                    ),
                    recommended_fix=(
                        "Reduce learning rate, check for reward scaling issues, "
                        "and apply gradient clipping."
                    ),
                )
        
        return None
    
    def reset(self) -> None:
                                   
        super().reset()
        self._q_values.clear()
        self._losses.clear()
        self._gradient_norms.clear()
        self._nan_detected = False
