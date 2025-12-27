   
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from .protocol import ExperimentProtocol, ExperimentResult
from .logger import ExperimentLogger
from ..environments import CareerEnv, EnvironmentFactory
from ..environments.param_engine import EnvironmentParams
from ..rewards import get_reward
from ..agents import DQNAgent, PPOAgent, RandomAgent
from ..agents.heuristic_agents import create_heuristic_agent, HEURISTIC_AGENTS
from ..failure_detection import (
    CombinedFailureDetector,
    RewardHackingDetector,
    PolicyCollapseDetector,
    NonstationarityDetector,
    ValueExplosionDetector,
)
from ..introspection import EntropyTracker, StateHeatmap, PhasePlotter
from ..utils.reproducibility import set_global_seed, get_reproducibility_info

class ExperimentRunner:
           
    AGENT_REGISTRY = {
        "dqn": DQNAgent,
        "ppo": PPOAgent,
        "random": RandomAgent,
    }
    
    def __init__(
        self,
        enable_failure_detection: bool = True,
        enable_introspection: bool = True,
        verbose: bool = True
    ):
                   
        self.enable_failure_detection = enable_failure_detection
        self.enable_introspection = enable_introspection
        self.verbose = verbose
    
    def run(self, protocol: ExperimentProtocol) -> ExperimentResult:
                   
        issues = protocol.validate()
        if issues:
            raise ValueError(f"Invalid protocol: {issues}")
        
        set_global_seed(protocol.seed)
        
        output_dir = Path(protocol.output_dir) / protocol.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        env = self._create_environment(protocol)
        reward_fn = self._create_reward(protocol)
        agent = self._create_agent(protocol, env)
        
        logger = ExperimentLogger(output_dir / "logs", protocol.log_to_tensorboard)
        failure_detector = self._create_failure_detector() if self.enable_failure_detection else None
        introspection = self._create_introspection() if self.enable_introspection else None
        
        logger.log_config(protocol.to_dict())
        logger.log_config({"reproducibility": get_reproducibility_info()})
        
        start_time = datetime.now()
        
        episode_rewards = []
        eval_rewards = []
        detected_failures = []
        
        iterator = range(protocol.n_episodes)
        if self.verbose:
            iterator = tqdm(iterator, desc="Training")
        
        for episode in iterator:
                                  
            ep_reward, ep_metrics = self._run_episode(
                env, agent, reward_fn, training=True, introspection=introspection
            )
            episode_rewards.append(ep_reward)
            
            logger.log_episode(episode, {
                "reward": ep_reward,
                **ep_metrics
            })
            
            if failure_detector is not None:
                failure_detector.update({
                    "step": episode,
                    "episode": episode,
                    "reward": ep_reward,
                    "episode_reward": ep_reward,
                    **ep_metrics
                })
                
                failures = failure_detector.detect_all()
                for f in failures:
                    detected_failures.append(f.to_dict())
                    if self.verbose:
                        print(f"\n⚠️ Failure detected: {f}")
            
            if (episode + 1) % protocol.eval_interval == 0:
                eval_reward = self._evaluate(
                    env, agent, reward_fn, protocol.eval_episodes
                )
                eval_rewards.append(eval_reward)
                logger.log_eval(episode, eval_reward)
                
                if self.verbose:
                    iterator.set_postfix({
                        "train": f"{np.mean(episode_rewards[-50:]):.2f}",
                        "eval": f"{eval_reward:.2f}"
                    })
            
            if (episode + 1) % protocol.checkpoint_interval == 0:
                agent.save(str(output_dir / f"checkpoint_ep{episode+1}.pt"))
        
        end_time = datetime.now()
        
        final_eval = self._evaluate(env, agent, reward_fn, protocol.eval_episodes * 2)
        
        if introspection is not None:
            self._save_introspection(introspection, output_dir)
        
        result = ExperimentResult(
            protocol=protocol,
            start_time=start_time,
            end_time=end_time,
            episode_rewards=episode_rewards,
            eval_rewards=eval_rewards,
            final_eval_reward=final_eval,
            best_eval_reward=max(eval_rewards) if eval_rewards else final_eval,
            detected_failures=detected_failures,
            agent_diagnostics=agent.get_diagnostics(),
            metrics={
                "train_reward_mean": np.mean(episode_rewards),
                "train_reward_std": np.std(episode_rewards),
                "eval_reward_mean": np.mean(eval_rewards) if eval_rewards else 0,
            }
        )
        
        result.save(str(output_dir / "result.json"))
        
        report = result.generate_report()
        (output_dir / "report.md").write_text(report)
        
        logger.close()
        
        if self.verbose:
            print(f"\n✅ Experiment complete!")
            print(f"   Final eval reward: {final_eval:.2f}")
            print(f"   Failures detected: {len(detected_failures)}")
            print(f"   Results saved to: {output_dir}")
        
        return result
    
    def _create_environment(self, protocol: ExperimentProtocol):
                                               
        params = EnvironmentParams(**protocol.env_params)
        return CareerEnv(params)
    
    def _create_reward(self, protocol: ExperimentProtocol):
                                                   
        return get_reward(protocol.reward_type, **protocol.reward_params)
    
    def _create_agent(self, protocol: ExperimentProtocol, env):
                                         
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        if protocol.agent_type in HEURISTIC_AGENTS:
            return create_heuristic_agent(
                protocol.agent_type, state_dim, action_dim, **protocol.agent_params
            )
        
        if protocol.agent_type not in self.AGENT_REGISTRY:
            raise ValueError(f"Unknown agent type: {protocol.agent_type}")
        
        agent_class = self.AGENT_REGISTRY[protocol.agent_type]
        return agent_class(state_dim, action_dim, **protocol.agent_params)
    
    def _create_failure_detector(self) -> CombinedFailureDetector:
                                                         
        return CombinedFailureDetector([
            RewardHackingDetector(),
            PolicyCollapseDetector(),
            NonstationarityDetector(),
            ValueExplosionDetector(),
        ])
    
    def _create_introspection(self) -> Dict[str, Any]:
                                         
        return {
            "entropy": EntropyTracker(),
            "heatmap": StateHeatmap(),
            "phase": PhasePlotter(),
        }
    
    def _run_episode(
        self,
        env,
        agent,
        reward_fn,
        training: bool = True,
        introspection: Optional[Dict] = None
    ) -> tuple:
                                   
        state, info = env.reset()
        reward_fn.reset()
        
        episode_reward = 0.0
        episode_states = [state.copy()]
        episode_actions = []
        episode_rewards = []
        
        done = False
        
        while not done:
                           
            action = agent.select_action(state, explore=training)
            episode_actions.append(action)
            
            next_state, base_reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            reward = reward_fn.compute(state, action, next_state, info, terminated, truncated)
            episode_reward += reward
            episode_rewards.append(reward)
            episode_states.append(next_state.copy())
            
            if training:
                metrics = agent.update({
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                })
            
            if introspection is not None:
                introspection["heatmap"].record(state[:2], action)
            
            state = next_state
        
        agent.on_episode_end()
        
        if introspection is not None:
            introspection["entropy"].update(agent.get_policy_entropy(), agent._episodes)
            introspection["phase"].add_trajectory(
                np.array(episode_states[:-1]),
                np.array(episode_actions),
                np.array(episode_rewards),
                agent._episodes
            )
        
        metrics = {
            "length": len(episode_actions),
            "final_burnout": info.get("burnout", 0),
            "final_salary": info.get("salary", 0),
            "entropy": agent.get_policy_entropy(),
        }
        
        return episode_reward, metrics
    
    def _evaluate(
        self,
        env,
        agent,
        reward_fn,
        n_episodes: int
    ) -> float:
                                         
        total_reward = 0.0
        
        for _ in range(n_episodes):
            state, _ = env.reset()
            reward_fn.reset()
            done = False
            
            while not done:
                action = agent.select_action(state, explore=False)
                next_state, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                reward = reward_fn.compute(state, action, next_state, info, terminated, truncated)
                total_reward += reward
                state = next_state
        
        return total_reward / n_episodes
    
    def _save_introspection(self, introspection: Dict, output_dir: Path) -> None:
                                         
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        introspection["entropy"].plot(
            save_path=str(viz_dir / "entropy.png"),
            show=False
        )
        
        introspection["heatmap"].plot(
            save_path=str(viz_dir / "state_heatmap.png"),
            show=False,
            include_actions=True,
            action_names=["STAY", "SWITCH"]
        )
        
        introspection["phase"].plot_multiple(
            n_trajectories=20,
            selection="last",
            save_path=str(viz_dir / "phase_plot.png"),
            show=False
        )

def run_experiment(
    name: str,
    **overrides
) -> ExperimentResult:
           
    protocol = ExperimentProtocol(name=name, **overrides)
    runner = ExperimentRunner()
    return runner.run(protocol)

if __name__ == "__main__":
          
    from .protocol import EXAMPLE_PROTOCOLS
    
    print("Running reward hacking study...")
    protocol = EXAMPLE_PROTOCOLS["reward_hacking_study"]
    protocol.n_episodes = 100              
    
    runner = ExperimentRunner()
    result = runner.run(protocol)
    
    print("\n" + result.generate_report())
