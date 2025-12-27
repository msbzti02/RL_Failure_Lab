   
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional

class ExperimentComparator:
           
    def __init__(self):
                                    
        self._results: Dict[str, Dict] = {}
    
    def add_result(self, path: str, name: Optional[str] = None) -> None:
                   
        with open(path, 'r') as f:
            result = json.load(f)
        
        if name is None:
            name = result.get("protocol", {}).get("name", Path(path).parent.name)
        
        self._results[name] = result
    
    def add_results_from_dir(self, directory: str) -> None:
                                               
        for result_file in Path(directory).glob("*/result.json"):
            self.add_result(str(result_file))
    
    def get_summary_table(self) -> pd.DataFrame:
                   
        rows = []
        
        for name, result in self._results.items():
            protocol = result.get("protocol", {})
            
            row = {
                "Experiment": name,
                "Agent": protocol.get("agent_type", "unknown"),
                "Reward": protocol.get("reward_type", "unknown"),
                "Episodes": protocol.get("n_episodes", 0),
                "Final Eval": result.get("final_eval_reward", 0),
                "Best Eval": result.get("best_eval_reward", 0),
                "Failures": len(result.get("detected_failures", [])),
                "Duration (s)": result.get("duration_seconds", 0),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def plot_learning_curves(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        smooth: int = 10
    ) -> None:
                   
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        for name, result in self._results.items():
            rewards = result.get("episode_rewards", [])
            if rewards:
                        
                if smooth > 1 and len(rewards) >= smooth:
                    kernel = np.ones(smooth) / smooth
                    smoothed = np.convolve(rewards, kernel, mode='valid')
                    episodes = np.arange(smooth - 1, len(rewards))
                else:
                    smoothed = rewards
                    episodes = np.arange(len(rewards))
                
                ax1.plot(episodes, smoothed, label=name, alpha=0.8)
        
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.set_title("Training Rewards")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        for name, result in self._results.items():
            eval_rewards = result.get("eval_rewards", [])
            protocol = result.get("protocol", {})
            eval_interval = protocol.get("eval_interval", 50)
            
            if eval_rewards:
                episodes = [(i + 1) * eval_interval for i in range(len(eval_rewards))]
                ax2.plot(episodes, eval_rewards, marker='o', label=name, alpha=0.8)
        
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Reward")
        ax2.set_title("Evaluation Rewards")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def plot_failure_comparison(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
                   
        failure_types = set()
        for result in self._results.values():
            for failure in result.get("detected_failures", []):
                failure_types.add(failure.get("failure_type", "unknown"))
        
        if not failure_types:
            print("No failures detected in any experiment")
            return
        
        data = {ftype: [] for ftype in failure_types}
        experiments = list(self._results.keys())
        
        for name in experiments:
            result = self._results[name]
            failures = result.get("detected_failures", [])
            
            type_counts = {}
            for f in failures:
                ftype = f.get("failure_type", "unknown")
                type_counts[ftype] = type_counts.get(ftype, 0) + 1
            
            for ftype in failure_types:
                data[ftype].append(type_counts.get(ftype, 0))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(experiments))
        width = 0.8 / len(failure_types)
        
        for i, ftype in enumerate(failure_types):
            offset = (i - len(failure_types) / 2 + 0.5) * width
            ax.bar(x + offset, data[ftype], width, label=ftype)
        
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Failure Count")
        ax.set_title("Failure Mode Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels(experiments, rotation=45, ha='right')
        ax.legend(title="Failure Type")
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def get_best_experiment(self, metric: str = "final_eval_reward") -> str:
                                                     
        best_name = None
        best_value = float('-inf')
        
        for name, result in self._results.items():
            value = result.get(metric, 0)
            if value > best_value:
                best_value = value
                best_name = name
        
        return best_name
    
    def generate_comparison_report(self) -> str:
                                                  
        lines = [
            "# Experiment Comparison Report",
            "",
            f"**Experiments compared:** {len(self._results)}",
            "",
            "## Summary Table",
            "",
            self.get_summary_table().to_markdown(index=False),
            "",
            "## Best Performer",
            "",
            f"**Best by final eval reward:** {self.get_best_experiment('final_eval_reward')}",
            "",
            "## Failure Analysis",
            "",
        ]
        
        for name, result in self._results.items():
            failures = result.get("detected_failures", [])
            if failures:
                lines.append(f"### {name}")
                for f in failures[:5]:           
                    lines.append(f"- [{f.get('severity', 'unknown').upper()}] "
                               f"{f.get('failure_type', 'unknown')}")
                if len(failures) > 5:
                    lines.append(f"- ... and {len(failures) - 5} more")
                lines.append("")
        
        return "\n".join(lines)

if __name__ == "__main__":
          
    print("ExperimentComparator demo")
    print("Usage: comparator.add_result('path/to/result.json')")
