   
from typing import Any, Dict, Optional, Tuple
import numpy as np
from gymnasium import spaces

from .base_env import BaseEnv
from .param_engine import EnvironmentParams, EnvironmentFactory

class CareerEnv(BaseEnv):
           
    STAY = 0
    SWITCH = 1
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self, 
        params: Optional[EnvironmentParams] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__(render_mode=render_mode)
        
        self.params = params or EnvironmentParams()
        
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(2)                  
        
        self._salary = 0.0
        self._burnout = 0.0
        self._tenure = 0
        self._market_condition = 0.0
        self._job_switches = 0
        
        self._episode_rewards = []
        self._episode_salaries = []
        self._episode_burnouts = []
        self._episode_actions = []
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
                                                 
        super().reset(seed=seed, options=options)
        
        self._salary = self.params.initial_salary
        self._burnout = self.params.initial_burnout
        self._tenure = 0
        self._job_switches = 0
        
        self._market_condition = self._sample_market_condition()
        
        self._episode_rewards = []
        self._episode_salaries = [self._salary]
        self._episode_burnouts = [self._burnout]
        self._episode_actions = []
        
        info = self._get_info()
        return self._get_obs(), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
                                                  
        super().step(action)
        
        self._episode_actions.append(action)
        
        prev_salary = self._salary
        prev_burnout = self._burnout
        
        if action == self.STAY:
            self._execute_stay()
        else:
            self._execute_switch()
        
        self._update_market()
        
        self._salary = np.clip(self._salary, self.params.min_salary, self.params.max_salary)
        self._burnout = np.clip(self._burnout, 0.0, self.params.burnout_threshold)
        
        self._episode_salaries.append(self._salary)
        self._episode_burnouts.append(self._burnout)
        
        terminated = self._burnout >= self.params.burnout_threshold
        truncated = self._current_step >= self.params.max_steps
        
        info = self._get_info()
        info["action_taken"] = "STAY" if action == self.STAY else "SWITCH"
        info["prev_salary"] = prev_salary
        info["prev_burnout"] = prev_burnout
        
        base_reward = self._compute_base_reward(prev_salary, prev_burnout)
        self._episode_rewards.append(base_reward)
        
        if terminated:
            info["termination_reason"] = "burnout"
        elif truncated:
            info["termination_reason"] = "max_steps"
        
        return self._get_obs(), base_reward, terminated, truncated, info
    
    def _execute_stay(self):
                                                       
        tenure_bonus = min(
            self._tenure * self.params.tenure_salary_bonus,
            self.params.max_tenure_bonus
        )
        growth_rate = self.params.salary_growth_mean + tenure_bonus
        
        market_factor = 1.0 + 0.5 * self._market_condition
        salary_change = self._np_random.normal(
            growth_rate,
            self.params.salary_volatility * market_factor
        )
        self._salary *= (1 + salary_change / 12)                      
        
        salary_ratio = self._salary / self.params.initial_salary
        burnout_increase = self.params.burnout_rate * (1 + 0.3 * (salary_ratio - 1))
        
        burnout_increase *= self._np_random.uniform(0.8, 1.2)
        self._burnout += burnout_increase
        
        self._tenure += 1
    
    def _execute_switch(self):
                                                 
        self._job_switches += 1
        
        switch_success = self._np_random.random() > self.params.switching_risk
        
        availability_factor = self.params.job_availability
        
        if switch_success:
                                              
            increase = self._np_random.uniform(0.05, 0.25) * availability_factor
            self._salary *= (1 + increase) * self.params.salary_multiplier
        else:
                                                         
            decrease = self._np_random.uniform(0.0, 0.15)
            self._salary *= (1 - decrease)
        
        recovery = self.params.burnout_recovery_rate * (1 + 0.5 * availability_factor)
        self._burnout = max(0, self._burnout - recovery - self._np_random.uniform(0.05, 0.15))
        
        self._tenure = 0
    
    def _update_market(self):
                                                       
        drift = -0.1 * self._market_condition                  
        noise = self._np_random.normal(0, 0.1)
        self._market_condition = np.clip(
            self._market_condition + drift + noise,
            -1.0, 1.0
        )
    
    def _sample_market_condition(self) -> float:
                                                              
        regime_means = {
            "stable": 0.0,
            "boom": 0.5,
            "recession": -0.5,
            "volatile": 0.0,
        }
        mean = regime_means.get(self.params.market_regime, 0.0)
        return np.clip(self._np_random.normal(mean, 0.2), -1.0, 1.0)
    
    def _compute_base_reward(self, prev_salary: float, prev_burnout: float) -> float:
                   
        salary_norm = (self._salary - self.params.min_salary) /                      (self.params.max_salary - self.params.min_salary)
        
        reward = salary_norm - self._burnout
        
        if self._burnout < 0.5:
            reward += 0.1
        
        return reward
    
    def _get_obs(self) -> np.ndarray:
                                         
        salary_norm = (self._salary - self.params.min_salary) /                      (self.params.max_salary - self.params.min_salary)
        tenure_norm = min(self._tenure / 100, 1.0)                    
        
        return np.array([
            salary_norm,
            self._burnout,
            tenure_norm,
            self._market_condition
        ], dtype=np.float32)
    
    def _get_info(self) -> Dict[str, Any]:
                                                             
        return {
            "salary": self._salary,
            "burnout": self._burnout,
            "tenure": self._tenure,
            "market_condition": self._market_condition,
            "job_switches": self._job_switches,
            "step": self._current_step,
            "episode": self._episode_count,
        }
    
    def get_config(self) -> Dict[str, Any]:
                                               
        return {
            "type": "career",
            "params": self.params.to_dict(),
        }
    
    def _render_ansi(self) -> str:
                                            
        burnout_bar = "█" * int(self._burnout * 20) + "░" * (20 - int(self._burnout * 20))
        market_indicator = "📈" if self._market_condition > 0.2 else                          "📉" if self._market_condition < -0.2 else "📊"
        
        return (
            f"╔══════════════════════════════════════╗\n"
            f"║ Step: {self._current_step:3d}/{self.params.max_steps:3d}  "
            f"Switches: {self._job_switches:2d}     ║\n"
            f"╠══════════════════════════════════════╣\n"
            f"║ 💰 Salary:  ${self._salary:,.0f}".ljust(40) + "║\n"
            f"║ 🔥 Burnout: [{burnout_bar}] {self._burnout:.0%}".ljust(42) + "║\n"
            f"║ 📅 Tenure:  {self._tenure} months".ljust(40) + "║\n"
            f"║ {market_indicator} Market:   {self._market_condition:+.2f}".ljust(39) + "║\n"
            f"╚══════════════════════════════════════╝"
        )
    
    def get_episode_summary(self) -> Dict[str, Any]:
                                                     
        if not self._episode_rewards:
            return {}
        
        return {
            "total_reward": sum(self._episode_rewards),
            "mean_reward": np.mean(self._episode_rewards),
            "final_salary": self._salary,
            "max_salary": max(self._episode_salaries),
            "final_burnout": self._burnout,
            "max_burnout": max(self._episode_burnouts),
            "job_switches": self._job_switches,
            "steps": self._current_step,
            "stay_ratio": 1 - (sum(self._episode_actions) / len(self._episode_actions))                          if self._episode_actions else 0,
        }

EnvironmentFactory.register("career", CareerEnv)

if __name__ == "__main__":
          
    from param_engine import EnvironmentParams
    
    print("Career Environment Demo\n")
    
    params = EnvironmentParams.for_regime("recession")
    env = CareerEnv(params)
    
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}\n")
    
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}: Action={'STAY' if action==0 else 'SWITCH'}")
        print(env.render(mode="ansi"))
        print(f"Reward: {reward:.3f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended: {'BURNOUT' if terminated else 'MAX STEPS'}")
            break
    
    print(f"\nEpisode Summary: {env.get_episode_summary()}")
