   
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

class FailureType(Enum):
                                           
    REWARD_HACKING = "reward_hacking"
    POLICY_COLLAPSE = "policy_collapse"
    NONSTATIONARITY = "nonstationarity"
    VALUE_EXPLOSION = "value_explosion"
    EXPLORATION_FAILURE = "exploration_failure"
    CREDIT_ASSIGNMENT = "credit_assignment"
    OVERFITTING = "overfitting"
    INSTABILITY = "instability"

@dataclass
class FailureReport:
                                                      
    failure_type: FailureType
    severity: str                                       
    timestamp: datetime
    step: int
    episode: int
    signal_values: Dict[str, float]
    description: str
    recommended_fix: str
    additional_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
                                                
        return {
            "failure_type": self.failure_type.value,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat(),
            "step": self.step,
            "episode": self.episode,
            "signal_values": self.signal_values,
            "description": self.description,
            "recommended_fix": self.recommended_fix,
            "additional_info": self.additional_info,
        }
    
    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.failure_type.value} at step {self.step}\n"
            f"  Description: {self.description}\n"
            f"  Fix: {self.recommended_fix}"
        )

class FailureTaxonomy:
           
    TAXONOMY = {
        FailureType.REWARD_HACKING: {
            "symptom": "High reward achieved but outcomes are poor",
            "root_cause": "Reward function has exploitable loopholes",
            "detection_signals": [
                "High reward + high burnout",
                "Reward increasing but policy behavior is degenerate",
                "Environment state degrades despite positive rewards"
            ],
            "mitigation": [
                "Redesign reward function",
                "Add constraints or penalties",
                "Use human evaluation to check outcomes"
            ],
            "example": "Agent learns to maximize salary bonus while ignoring burnout penalties"
        },
        
        FailureType.POLICY_COLLAPSE: {
            "symptom": "Agent always takes the same action",
            "root_cause": "Exploration decays too fast or entropy drops",
            "detection_signals": [
                "Action entropy approaching 0",
                "Single action dominance in distribution",
                "Value estimates become identical for all actions"
            ],
            "mitigation": [
                "Increase entropy coefficient",
                "Slow down epsilon decay",
                "Add exploration bonuses"
            ],
            "example": "Agent learns to always STAY regardless of state"
        },
        
        FailureType.NONSTATIONARITY: {
            "symptom": "Oscillating training curves or unstable performance",
            "root_cause": "Environment dynamics change due to agent's actions",
            "detection_signals": [
                "High reward variance over time",
                "Performance cycling up and down",
                "Policy changes don't lead to stable improvement"
            ],
            "mitigation": [
                "Use curriculum learning",
                "Add temporal regularization",
                "Reduce learning rate"
            ],
            "example": "Agent's switching behavior affects market conditions which affects agent"
        },
        
        FailureType.VALUE_EXPLOSION: {
            "symptom": "NaN values, divergent training, extreme Q-values",
            "root_cause": "Poor scaling or unstable learning dynamics",
            "detection_signals": [
                "Q-values exceeding reasonable bounds",
                "Loss becomes NaN or Inf",
                "Gradient norms exploding"
            ],
            "mitigation": [
                "Apply gradient clipping",
                "Reduce learning rate",
                "Normalize rewards",
                "Check for bootstrap issues"
            ],
            "example": "Q-values grow unboundedly in DQN training"
        },
        
        FailureType.EXPLORATION_FAILURE: {
            "symptom": "Agent converges to suboptimal policy early",
            "root_cause": "Insufficient exploration of state-action space",
            "detection_signals": [
                "Low state space coverage",
                "Converges before visiting important states",
                "Performance plateaus at suboptimal level"
            ],
            "mitigation": [
                "Increase initial exploration",
                "Use intrinsic motivation",
                "Implement curiosity-driven exploration"
            ],
            "example": "Agent never tries SWITCH action and misses salary increases"
        },
        
        FailureType.CREDIT_ASSIGNMENT: {
            "symptom": "Agent fails to learn despite clear optimal actions",
            "root_cause": "Delayed rewards make it hard to assign credit",
            "detection_signals": [
                "Slow or no learning progress",
                "High variance in value estimates",
                "Random-like behavior persists"
            ],
            "mitigation": [
                "Add reward shaping",
                "Use eligibility traces",
                "Reduce delay in rewards"
            ],
            "example": "With delayed rewards, agent can't attribute final outcome to early switches"
        },
        
        FailureType.OVERFITTING: {
            "symptom": "Good training performance, poor generalization",
            "root_cause": "Agent memorizes training conditions",
            "detection_signals": [
                "Gap between train and eval performance",
                "Poor performance on parameter variations",
                "Policy is overly specific to seen states"
            ],
            "mitigation": [
                "Add regularization",
                "Use domain randomization",
                "Increase environment variety"
            ],
            "example": "Agent performs well in stable market but fails in recession"
        },
        
        FailureType.INSTABILITY: {
            "symptom": "Erratic learning curves, performance collapses",
            "root_cause": "Hyperparameter sensitivity or algorithmic instability",
            "detection_signals": [
                "Sharp drops in performance",
                "High variance in training metrics",
                "Sensitivity to random seeds"
            ],
            "mitigation": [
                "Tune hyperparameters carefully",
                "Add stabilization mechanisms",
                "Use multiple seeds for evaluation"
            ],
            "example": "Performance drops suddenly after 500 episodes"
        },
    }
    
    @classmethod
    def get_all(cls) -> Dict[FailureType, Dict]:
                                    
        return cls.TAXONOMY.copy()
    
    @classmethod
    def get_failure_info(cls, failure_type: FailureType) -> Dict:
                                                            
        return cls.TAXONOMY.get(failure_type, {})
    
    @classmethod
    def get_mitigation(cls, failure_type: FailureType) -> List[str]:
                                                           
        return cls.TAXONOMY.get(failure_type, {}).get("mitigation", [])
    
    @classmethod
    def to_markdown_table(cls) -> str:
                                                      
        lines = [
            "| Failure Type | Symptom | Root Cause | Detection Signal |",
            "|--------------|---------|------------|------------------|",
        ]
        
        for ftype, info in cls.TAXONOMY.items():
            symptom = info.get("symptom", "N/A")
            cause = info.get("root_cause", "N/A")
            signals = info.get("detection_signals", ["N/A"])
            signal = signals[0] if signals else "N/A"
            
            lines.append(f"| {ftype.value} | {symptom} | {cause} | {signal} |")
        
        return "\n".join(lines)
    
    @classmethod
    def generate_playbook(cls) -> str:
                                                    
        sections = ["# RL Failure Diagnosis Playbook\n"]
        
        for ftype, info in cls.TAXONOMY.items():
            section = f"\n## {ftype.value.replace('_', ' ').title()}\n\n"
            section += f"**Symptom:** {info.get('symptom', 'N/A')}\n\n"
            section += f"**Root Cause:** {info.get('root_cause', 'N/A')}\n\n"
            
            section += "**Detection Signals:**\n"
            for signal in info.get("detection_signals", []):
                section += f"- {signal}\n"
            
            section += "\n**Recommended Fixes:**\n"
            for fix in info.get("mitigation", []):
                section += f"1. {fix}\n"
            
            if "example" in info:
                section += f"\n**Example:** {info['example']}\n"
            
            sections.append(section)
        
        return "".join(sections)

if __name__ == "__main__":
          
    print("Failure Taxonomy Table:")
    print(FailureTaxonomy.to_markdown_table())
    
    print("\n\nPlaybook Preview:")
    print(FailureTaxonomy.generate_playbook()[:1000] + "...")
