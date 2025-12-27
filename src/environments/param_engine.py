   
from dataclasses import dataclass
from typing import Dict, Any, Optional, Type
from enum import Enum

import numpy as np

class MarketRegime(Enum):
                                       
    STABLE = "stable"
    BOOM = "boom"
    RECESSION = "recession"
    VOLATILE = "volatile"

MARKET_REGIMES: Dict[str, Dict[str, Any]] = {
    "stable": {
        "salary_growth_mean": 0.03,                        
        "salary_volatility": 0.10,                        
        "burnout_rate": 0.08,                               
        "switching_risk": 0.25,                     
        "job_availability": 0.8,                             
        "salary_multiplier": 1.0,                          
    },
    "boom": {
        "salary_growth_mean": 0.08,                  
        "salary_volatility": 0.15,                             
        "burnout_rate": 0.12,                                        
        "switching_risk": 0.15,                           
        "job_availability": 0.95,                        
        "salary_multiplier": 1.3,                          
    },
    "recession": {
        "salary_growth_mean": -0.02,                       
        "salary_volatility": 0.25,                         
        "burnout_rate": 0.15,                                    
        "switching_risk": 0.50,                            
        "job_availability": 0.4,                    
        "salary_multiplier": 0.85,                        
    },
    "volatile": {
        "salary_growth_mean": 0.02,                   
        "salary_volatility": 0.35,                              
        "burnout_rate": 0.10,                               
        "switching_risk": 0.35,                          
        "job_availability": 0.6,                                  
        "salary_multiplier": 1.1,                                   
    },
}

@dataclass
class EnvironmentParams:
           
    burnout_rate: float = 0.1                                                        
    salary_volatility: float = 0.15                                         
    switching_risk: float = 0.3                                               
    
    market_regime: str = "stable"                                          
    salary_growth_mean: float = 0.03                                          
    job_availability: float = 0.8                                          
    salary_multiplier: float = 1.0                                          
    
    initial_salary: float = 50000.0                      
    initial_burnout: float = 0.0                                 
    min_salary: float = 30000.0                       
    max_salary: float = 300000.0                        
    
    max_steps: int = 200                                            
    
    burnout_recovery_rate: float = 0.02                                
    burnout_threshold: float = 1.0                               
    
    tenure_salary_bonus: float = 0.01                                       
    max_tenure_bonus: float = 0.15                            
    
    def __post_init__(self):
                                                                   
        self._apply_market_regime()
        self._validate()
    
    def _apply_market_regime(self):
                                                         
        if self.market_regime in MARKET_REGIMES:
            regime = MARKET_REGIMES[self.market_regime]
                                             
            for key, value in regime.items():
                if hasattr(self, key):
                                                          
                    default = EnvironmentParams.__dataclass_fields__[key].default
                    if getattr(self, key) == default:
                        setattr(self, key, value)
    
    def _validate(self):
                                        
        validations = [
            (0.01 <= self.burnout_rate <= 0.5, "burnout_rate must be in [0.01, 0.5]"),
            (0.0 <= self.salary_volatility <= 0.5, "salary_volatility must be in [0.0, 0.5]"),
            (0.0 <= self.switching_risk <= 1.0, "switching_risk must be in [0.0, 1.0]"),
            (50 <= self.max_steps <= 500, "max_steps must be in [50, 500]"),
            (0.0 <= self.initial_burnout <= 1.0, "initial_burnout must be in [0.0, 1.0]"),
            (self.min_salary < self.max_salary, "min_salary must be < max_salary"),
            (self.initial_salary >= self.min_salary, "initial_salary must be >= min_salary"),
        ]
        
        for condition, message in validations:
            if not condition:
                raise ValueError(f"Invalid parameter: {message}")
    
    def to_dict(self) -> Dict[str, Any]:
                                                           
        return {
            "burnout_rate": self.burnout_rate,
            "salary_volatility": self.salary_volatility,
            "switching_risk": self.switching_risk,
            "market_regime": self.market_regime,
            "salary_growth_mean": self.salary_growth_mean,
            "job_availability": self.job_availability,
            "salary_multiplier": self.salary_multiplier,
            "initial_salary": self.initial_salary,
            "initial_burnout": self.initial_burnout,
            "max_steps": self.max_steps,
            "burnout_recovery_rate": self.burnout_recovery_rate,
            "burnout_threshold": self.burnout_threshold,
            "tenure_salary_bonus": self.tenure_salary_bonus,
            "max_tenure_bonus": self.max_tenure_bonus,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnvironmentParams":
                                                       
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    @classmethod
    def for_regime(cls, regime: str, **overrides) -> "EnvironmentParams":
                                                                                     
        params = cls(market_regime=regime)
        for key, value in overrides.items():
            if hasattr(params, key):
                setattr(params, key, value)
        params._validate()
        return params

class EnvironmentFactory:
           
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, env_class: Type) -> None:
                                            
        cls._registry[name] = env_class
    
    @classmethod
    def create(cls, env_type: str, **kwargs) -> Any:
                   
        if env_type not in cls._registry:
            raise ValueError(f"Unknown environment type: {env_type}. "
                           f"Available: {list(cls._registry.keys())}")
        
        env_class = cls._registry[env_type]
        params = EnvironmentParams(**kwargs) if kwargs else EnvironmentParams()
        return env_class(params)
    
    @classmethod
    def create_from_config(cls, env_config) -> Any:
                                                          
        return cls.create(
            env_config.type,
            burnout_rate=env_config.burnout_rate,
            salary_volatility=env_config.salary_volatility,
            switching_risk=env_config.switching_risk,
            market_regime=env_config.market_regime,
            max_steps=env_config.max_steps,
            initial_salary=env_config.initial_salary,
            initial_burnout=env_config.initial_burnout,
        )
    
    @classmethod
    def list_environments(cls) -> list:
                                                    
        return list(cls._registry.keys())
    
    @classmethod
    def list_regimes(cls) -> list:
                                                
        return list(MARKET_REGIMES.keys())

def stress_test_params() -> Dict[str, EnvironmentParams]:
           
    return {
        "high_volatility": EnvironmentParams(
            salary_volatility=0.4,
            market_regime="volatile"
        ),
        "rapid_burnout": EnvironmentParams(
            burnout_rate=0.3,
            burnout_recovery_rate=0.01
        ),
        "risky_switching": EnvironmentParams(
            switching_risk=0.7,
            job_availability=0.3
        ),
        "long_horizon": EnvironmentParams(
            max_steps=500,
            burnout_rate=0.05
        ),
        "harsh_recession": EnvironmentParams(
            market_regime="recession",
            switching_risk=0.6,
            job_availability=0.2
        ),
    }

if __name__ == "__main__":
          
    print("Available market regimes:")
    for regime, params in MARKET_REGIMES.items():
        print(f"\n{regime.upper()}:")
        for k, v in params.items():
            print(f"  {k}: {v}")
    
    print("\n\nStress test configurations:")
    for name, params in stress_test_params().items():
        print(f"\n{name}:")
        for k, v in list(params.to_dict().items())[:5]:
            print(f"  {k}: {v}")
