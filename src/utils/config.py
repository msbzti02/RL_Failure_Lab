   
import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class EnvConfig:
                                    
    type: str = "career"
    burnout_rate: float = 0.1
    salary_volatility: float = 0.15
    switching_risk: float = 0.3
    market_regime: str = "stable"                           
    max_steps: int = 200
    initial_salary: float = 50000.0
    initial_burnout: float = 0.0

@dataclass
class RewardConfig:
                                        
    type: str = "short_term_v1"
    params: dict = field(default_factory=dict)

@dataclass
class AgentConfig:
                              
    type: str = "dqn"
    learning_rate: float = 0.001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 64
    hidden_dims: list = field(default_factory=lambda: [64, 64])
    target_update_freq: int = 100
                  
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    gae_lambda: float = 0.95
    n_epochs: int = 10
    n_steps: int = 2048

@dataclass
class TrainingConfig:
                                 
    episodes: int = 1000
    eval_interval: int = 50
    eval_episodes: int = 10
    checkpoint_interval: int = 100
    log_interval: int = 10

@dataclass
class Config:
                                            
    name: str = "default_experiment"
    seed: int = 42
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output_dir: str = "experiments/results"
    
    def to_dict(self) -> dict:
                                                       
        return {
            "name": self.name,
            "seed": self.seed,
            "env": self.env.__dict__,
            "reward": {"type": self.reward.type, "params": self.reward.params},
            "agent": self.agent.__dict__,
            "training": self.training.__dict__,
            "output_dir": self.output_dir,
        }
    
    def save(self, path: str) -> None:
                                              
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
                                            
        config = cls()
        config.name = d.get("name", config.name)
        config.seed = d.get("seed", config.seed)
        config.output_dir = d.get("output_dir", config.output_dir)
        
        if "env" in d:
            for k, v in d["env"].items():
                if hasattr(config.env, k):
                    setattr(config.env, k, v)
        
        if "reward" in d:
            config.reward.type = d["reward"].get("type", config.reward.type)
            config.reward.params = d["reward"].get("params", {})
        
        if "agent" in d:
            for k, v in d["agent"].items():
                if hasattr(config.agent, k):
                    setattr(config.agent, k, v)
        
        if "training" in d:
            for k, v in d["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        
        return config

def load_config(path: str) -> Config:
           
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    env_overrides = {
        "RL_LAB_SEED": ("seed", int),
        "RL_LAB_EPISODES": ("training.episodes", int),
        "RL_LAB_LEARNING_RATE": ("agent.learning_rate", float),
    }
    
    for env_var, (key_path, type_fn) in env_overrides.items():
        if env_var in os.environ:
            value = type_fn(os.environ[env_var])
            keys = key_path.split(".")
            d = data
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    
    return Config.from_dict(data)

def create_default_configs():
                                                                     
    configs_dir = Path("configs")
    
    default = Config()
    default.save(configs_dir / "experiments" / "default.yaml")
    
    reward_hacking = Config(
        name="reward_hacking_study",
        env=EnvConfig(burnout_rate=0.05, salary_volatility=0.25),
        reward=RewardConfig(type="short_term_v1"),
        training=TrainingConfig(episodes=500),
    )
    reward_hacking.save(configs_dir / "experiments" / "reward_hacking_study.yaml")
    
    collapse_study = Config(
        name="policy_collapse_study",
        agent=AgentConfig(epsilon_decay=0.99, entropy_coef=0.001),
        training=TrainingConfig(episodes=1000),
    )
    collapse_study.save(configs_dir / "experiments" / "policy_collapse_study.yaml")

if __name__ == "__main__":
    create_default_configs()
    print("Created default configuration files.")
