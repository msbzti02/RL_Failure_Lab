   
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import yaml
import json
from pathlib import Path

@dataclass
class ExperimentProtocol:
           
    name: str
    description: str = ""
    version: str = "1.0"
    
    seed: int = 42
    
    env_type: str = "career"
    env_params: Dict[str, Any] = field(default_factory=dict)
    
    reward_type: str = "short_term_v1"
    reward_params: Dict[str, Any] = field(default_factory=dict)
    
    agent_type: str = "dqn"
    agent_params: Dict[str, Any] = field(default_factory=dict)
    
    n_episodes: int = 1000
    eval_interval: int = 50
    eval_episodes: int = 10
    checkpoint_interval: int = 100
    
    hypothesis: str = ""
    expected_failure: str = ""
    
    output_dir: str = "experiments/results"
    log_to_tensorboard: bool = True
    save_trajectories: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
                                    
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "seed": self.seed,
            "env_type": self.env_type,
            "env_params": self.env_params,
            "reward_type": self.reward_type,
            "reward_params": self.reward_params,
            "agent_type": self.agent_type,
            "agent_params": self.agent_params,
            "n_episodes": self.n_episodes,
            "eval_interval": self.eval_interval,
            "eval_episodes": self.eval_episodes,
            "checkpoint_interval": self.checkpoint_interval,
            "hypothesis": self.hypothesis,
            "expected_failure": self.expected_failure,
            "output_dir": self.output_dir,
            "log_to_tensorboard": self.log_to_tensorboard,
            "save_trajectories": self.save_trajectories,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentProtocol":
                                     
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def save(self, path: str) -> None:
                                         
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentProtocol":
                                           
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)
    
    def validate(self) -> List[str]:
                                                          
        issues = []
        
        if not self.name:
            issues.append("Experiment name is required")
        
        if self.seed < 0:
            issues.append("Seed must be non-negative")
        
        if self.n_episodes <= 0:
            issues.append("n_episodes must be positive")
        
        if self.eval_interval <= 0:
            issues.append("eval_interval must be positive")
        
        return issues

@dataclass
class ExperimentResult:
           
    protocol: ExperimentProtocol
    
    start_time: datetime
    end_time: datetime
    
    episode_rewards: List[float] = field(default_factory=list)
    eval_rewards: List[float] = field(default_factory=list)
    
    final_eval_reward: float = 0.0
    best_eval_reward: float = 0.0
    
    detected_failures: List[Dict[str, Any]] = field(default_factory=list)
    
    agent_diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    observed_failure: str = ""
    fix_applied: str = ""
    outcome: str = ""
    
    @property
    def duration_seconds(self) -> float:
                                             
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success(self) -> bool:
                                                                     
        critical_failures = [f for f in self.detected_failures 
                           if f.get("severity") == "critical"]
        return len(critical_failures) == 0
    
    def to_dict(self) -> Dict[str, Any]:
                                    
        return {
            "protocol": self.protocol.to_dict(),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": self.duration_seconds,
            "episode_rewards": self.episode_rewards,
            "eval_rewards": self.eval_rewards,
            "final_eval_reward": self.final_eval_reward,
            "best_eval_reward": self.best_eval_reward,
            "detected_failures": self.detected_failures,
            "agent_diagnostics": self.agent_diagnostics,
            "metrics": self.metrics,
            "observed_failure": self.observed_failure,
            "fix_applied": self.fix_applied,
            "outcome": self.outcome,
            "success": self.success,
        }
    
    def save(self, path: str) -> None:
                                        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def generate_report(self) -> str:
                                                  
        report = [
            f"# Experiment Report: {self.protocol.name}",
            "",
            f"**Date:** {self.start_time.strftime('%Y-%m-%d %H:%M')}",
            f"**Duration:** {self.duration_seconds:.1f} seconds",
            f"**Status:** {'✅ Success' if self.success else '❌ Failed'}",
            "",
            "## Hypothesis",
            self.protocol.hypothesis or "_Not specified_",
            "",
            "## Expected Failure",
            self.protocol.expected_failure or "_Not specified_",
            "",
            "## Configuration",
            f"- **Seed:** {self.protocol.seed}",
            f"- **Environment:** {self.protocol.env_type}",
            f"- **Reward:** {self.protocol.reward_type}",
            f"- **Agent:** {self.protocol.agent_type}",
            f"- **Episodes:** {self.protocol.n_episodes}",
            "",
            "## Results",
            f"- **Final Eval Reward:** {self.final_eval_reward:.2f}",
            f"- **Best Eval Reward:** {self.best_eval_reward:.2f}",
            "",
            "## Detected Failures",
        ]
        
        if self.detected_failures:
            for failure in self.detected_failures:
                report.append(f"- [{failure.get('severity', 'unknown').upper()}] "
                            f"{failure.get('failure_type', 'unknown')}: "
                            f"{failure.get('description', '')[:100]}")
        else:
            report.append("_No failures detected_")
        
        report.extend([
            "",
            "## Observed Outcome",
            self.observed_failure or "_To be filled_",
            "",
            "## Fix Applied",
            self.fix_applied or "_To be filled_",
            "",
            "## Final Outcome",
            self.outcome or "_To be filled_",
        ])
        
        return "\n".join(report)

EXAMPLE_PROTOCOLS = {
    "reward_hacking_study": ExperimentProtocol(
        name="reward_hacking_study",
        description="Study reward hacking with short-term reward",
        hypothesis="Agent will exploit salary jumps while ignoring burnout",
        expected_failure="reward_hacking",
        env_params={"burnout_rate": 0.05, "salary_volatility": 0.25},
        reward_type="short_term_v1",
        reward_params={"lambda_burnout": 0.5},                       
        n_episodes=500,
    ),
    
    "policy_collapse_study": ExperimentProtocol(
        name="policy_collapse_study",
        description="Study policy collapse with fast epsilon decay",
        hypothesis="Fast epsilon decay will cause policy collapse to STAY",
        expected_failure="policy_collapse",
        agent_type="dqn",
        agent_params={"epsilon_decay": 0.99, "epsilon_end": 0.0},              
        n_episodes=500,
    ),
    
    "sparse_reward_challenge": ExperimentProtocol(
        name="sparse_reward_challenge",
        description="Test learning with sparse rewards",
        hypothesis="Sparse rewards will cause exploration failure",
        expected_failure="exploration_failure",
        reward_type="sparse_v1",
        n_episodes=1000,
    ),
    
    "recession_stress_test": ExperimentProtocol(
        name="recession_stress_test",
        description="Test agent performance in harsh recession",
        hypothesis="Agent will fail to generalize to recession conditions",
        expected_failure="nonstationarity",
        env_params={"market_regime": "recession"},
        n_episodes=500,
    ),
}
