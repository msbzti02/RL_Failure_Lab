   
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import copy

@dataclass
class CounterfactualResult:
                                             
    original_action: int
    original_trajectory: List[Dict[str, Any]]
    original_return: float
    
    alternative_action: int
    alternative_trajectory: List[Dict[str, Any]]
    alternative_return: float
    
    @property
    def return_difference(self) -> float:
                                                         
        return self.alternative_return - self.original_return
    
    @property
    def was_better_choice(self) -> bool:
                                                             
        return self.alternative_return > self.original_return
    
    def summary(self) -> str:
                                              
        action_names = ["STAY", "SWITCH"]
        diff = self.return_difference
        
        if abs(diff) < 0.1:
            verdict = "roughly equivalent"
        elif diff > 0:
            verdict = f"would have been better by {diff:.2f}"
        else:
            verdict = f"would have been worse by {-diff:.2f}"
        
        return (
            f"Original: {action_names[self.original_action]} → Return: {self.original_return:.2f}\n"
            f"Alternative: {action_names[self.alternative_action]} → Return: {self.alternative_return:.2f}\n"
            f"Verdict: {action_names[self.alternative_action]} {verdict}"
        )

class CounterfactualAnalyzer:
           
    def __init__(
        self,
        env,
        agent,
        reward_fn: Optional[Callable] = None,
        gamma: float = 0.99
    ):
                   
        self.env = env
        self.agent = agent
        self.reward_fn = reward_fn
        self.gamma = gamma
    
    def analyze(
        self,
        state: np.ndarray,
        original_action: int,
        horizon: int = 10,
        n_rollouts: int = 5
    ) -> CounterfactualResult:
                   
        alternative_action = 1 - original_action                      
        
        original_trajectories = []
        for _ in range(n_rollouts):
            traj = self._rollout(state, original_action, horizon)
            original_trajectories.append(traj)
        
        original_return = np.mean([
            self._compute_return(t) for t in original_trajectories
        ])
        
        alternative_trajectories = []
        for _ in range(n_rollouts):
            traj = self._rollout(state, alternative_action, horizon)
            alternative_trajectories.append(traj)
        
        alternative_return = np.mean([
            self._compute_return(t) for t in alternative_trajectories
        ])
        
        return CounterfactualResult(
            original_action=original_action,
            original_trajectory=original_trajectories[0],
            original_return=original_return,
            alternative_action=alternative_action,
            alternative_trajectory=alternative_trajectories[0],
            alternative_return=alternative_return,
        )
    
    def _rollout(
        self,
        initial_state: np.ndarray,
        first_action: int,
        horizon: int
    ) -> List[Dict[str, Any]]:
                                                                
        trajectory = []
        
        env = copy.deepcopy(self.env)
        
        state = initial_state.copy()
        done = False
        
        for step in range(horizon):
            if done:
                break
            
            if step == 0:
                action = first_action
            else:
                action = self.agent.select_action(state, explore=False)
            
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except:
                                             
                next_state = state.copy()
                reward = 0.0
                done = step >= horizon - 1
                info = {}
            
            if self.reward_fn is not None:
                reward = self.reward_fn(state, action, next_state, info, terminated, truncated)
            
            trajectory.append({
                "state": state.copy(),
                "action": action,
                "reward": reward,
                "next_state": next_state.copy(),
                "done": done,
                "info": info,
            })
            
            state = next_state
        
        return trajectory
    
    def _compute_return(self, trajectory: List[Dict[str, Any]]) -> float:
                                                        
        G = 0.0
        for i, step in enumerate(reversed(trajectory)):
            G = step["reward"] + self.gamma * G
        return G
    
    def batch_analyze(
        self,
        states: List[np.ndarray],
        actions: List[int],
        horizon: int = 10,
        n_rollouts: int = 3
    ) -> List[CounterfactualResult]:
                                               
        results = []
        for state, action in zip(states, actions):
            result = self.analyze(state, action, horizon, n_rollouts)
            results.append(result)
        return results
    
    def get_regret_summary(self, results: List[CounterfactualResult]) -> Dict[str, Any]:
                   
        regrets = [max(0, r.return_difference) for r in results]
        wrong_choices = sum(1 for r in results if r.was_better_choice)
        
        return {
            "total_regret": sum(regrets),
            "mean_regret": np.mean(regrets),
            "max_regret": max(regrets) if regrets else 0,
            "wrong_choice_rate": wrong_choices / len(results) if results else 0,
            "n_decisions": len(results),
        }
