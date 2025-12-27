   
from typing import Dict, Type, Optional, List
from .base_reward import BaseReward

class RewardRegistry:
           
    _rewards: Dict[str, Type[BaseReward]] = {}
    
    @classmethod
    def register(cls, name: str, reward_class: Type[BaseReward]) -> None:
                   
        if not issubclass(reward_class, BaseReward):
            raise TypeError(f"{reward_class} must be a subclass of BaseReward")
        cls._rewards[name] = reward_class
    
    @classmethod
    def get(cls, name: str, **params) -> BaseReward:
                   
        if name not in cls._rewards:
            available = list(cls._rewards.keys())
            raise ValueError(f"Unknown reward function: {name}. Available: {available}")
        
        return cls._rewards[name](**params)
    
    @classmethod
    def list(cls) -> List[str]:
                                                   
        return list(cls._rewards.keys())
    
    @classmethod
    def get_documentation(cls, name: Optional[str] = None) -> Dict:
                   
        if name:
            if name not in cls._rewards:
                raise ValueError(f"Unknown reward function: {name}")
            return cls._rewards[name].get_documentation()
        
        return {
            name: reward_cls.get_documentation()
            for name, reward_cls in cls._rewards.items()
        }
    
    @classmethod
    def get_comparison_table(cls) -> str:
                                                                           
        lines = [
            "| Reward Type | Intended Behavior | Known Failure Modes |",
            "|-------------|-------------------|---------------------|",
        ]
        
        for name, reward_cls in cls._rewards.items():
            doc = reward_cls.get_documentation()
            behavior = doc.get("intended_behavior", "N/A")
            failures = ", ".join(doc.get("known_failure_modes", ["N/A"]))
            lines.append(f"| {name} | {behavior} | {failures} |")
        
        return "\n".join(lines)

def get_reward(name: str, **params) -> BaseReward:
                                                  
    return RewardRegistry.get(name, **params)

def list_rewards() -> List[str]:
                                              
    return RewardRegistry.list()

def register_reward(name: str):
                                                  
    def decorator(cls: Type[BaseReward]) -> Type[BaseReward]:
        RewardRegistry.register(name, cls)
        return cls
    return decorator
