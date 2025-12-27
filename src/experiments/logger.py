   
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

class ExperimentLogger:
           
    def __init__(
        self,
        log_dir: Path,
        use_tensorboard: bool = True
    ):
                   
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_tensorboard = use_tensorboard
        self._writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(str(self.log_dir / "tensorboard"))
            except ImportError:
                print("TensorBoard not available, disabling")
                self.use_tensorboard = False
        
        self._log_file = open(self.log_dir / "training.jsonl", 'w')
        
        self._episode_metrics = []
        self._eval_metrics = []
    
    def log_config(self, config: Dict[str, Any]) -> None:
                                           
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]) -> None:
                                  
        entry = {
            "type": "episode",
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            **metrics
        }
        
        self._episode_metrics.append(entry)
        self._log_file.write(json.dumps(entry) + "\n")
        self._log_file.flush()
        
        if self._writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f"train/{key}", value, episode)
    
    def log_eval(self, episode: int, reward: float) -> None:
                                     
        entry = {
            "type": "eval",
            "episode": episode,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
        }
        
        self._eval_metrics.append(entry)
        self._log_file.write(json.dumps(entry) + "\n")
        self._log_file.flush()
        
        if self._writer:
            self._writer.add_scalar("eval/reward", reward, episode)
    
    def log_failure(self, episode: int, failure: Dict[str, Any]) -> None:
                                   
        entry = {
            "type": "failure",
            "episode": episode,
            "timestamp": datetime.now().isoformat(),
            **failure
        }
        
        self._log_file.write(json.dumps(entry) + "\n")
        self._log_file.flush()
        
        if self._writer:
            self._writer.add_text(
                "failures",
                f"Episode {episode}: {failure.get('failure_type', 'unknown')}",
                episode
            )
    
    def log_metrics(self, step: int, metrics: Dict[str, Any], prefix: str = "") -> None:
                                    
        if self._writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    tag = f"{prefix}/{key}" if prefix else key
                    self._writer.add_scalar(tag, value, step)
    
    def close(self) -> None:
                                              
        self._log_file.close()
        
        if self._writer:
            self._writer.close()
        
        summary = {
            "total_episodes": len(self._episode_metrics),
            "total_evals": len(self._eval_metrics),
        }
        
        if self._episode_metrics:
            rewards = [e.get("reward", 0) for e in self._episode_metrics]
            summary["final_reward_mean"] = sum(rewards[-100:]) / min(100, len(rewards))
        
        with open(self.log_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
