   
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from collections import defaultdict

class StateHeatmap:
           
    def __init__(
        self,
        bins: Tuple[int, int] = (20, 20),
        x_range: Tuple[float, float] = (0, 1),
        y_range: Tuple[float, float] = (0, 1),
        x_label: str = "Salary (normalized)",
        y_label: str = "Burnout"
    ):
                   
        self.bins = bins
        self.x_range = x_range
        self.y_range = y_range
        self.x_label = x_label
        self.y_label = y_label
        
        self._visit_counts = np.zeros(bins)
        self._state_actions = defaultdict(lambda: defaultdict(int))
        self._total_visits = 0
    
    def record(self, state_2d: np.ndarray, action: Optional[int] = None) -> None:
                   
        x, y = state_2d[0], state_2d[1]
        
        x_bin = int((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * self.bins[0])
        y_bin = int((y - self.y_range[0]) / (self.y_range[1] - self.y_range[0]) * self.bins[1])
        
        x_bin = max(0, min(x_bin, self.bins[0] - 1))
        y_bin = max(0, min(y_bin, self.bins[1] - 1))
        
        self._visit_counts[x_bin, y_bin] += 1
        self._total_visits += 1
        
        if action is not None:
            self._state_actions[(x_bin, y_bin)][action] += 1
    
    def get_visit_frequency(self) -> np.ndarray:
                                             
        if self._total_visits == 0:
            return self._visit_counts
        return self._visit_counts / self._total_visits
    
    def get_unvisited_states(self) -> float:
                                                 
        return np.sum(self._visit_counts == 0) / np.prod(self.bins)
    
    def get_coverage(self) -> float:
                                                         
        return 1 - self.get_unvisited_states()
    
    def get_action_heatmap(self, action: int) -> np.ndarray:
                                                               
        action_map = np.zeros(self.bins)
        for (x_bin, y_bin), actions in self._state_actions.items():
            if action in actions:
                total = sum(actions.values())
                action_map[x_bin, y_bin] = actions[action] / total
        return action_map
    
    def plot(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        include_actions: bool = False,
        action_names: List[str] = None
    ) -> None:
                   
        if include_actions and self._state_actions:
            n_actions = max(max(actions.keys()) for actions in self._state_actions.values()) + 1
            fig, axes = plt.subplots(1, n_actions + 1, figsize=(5 * (n_actions + 1), 4))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes = [axes]
        
        freq = self.get_visit_frequency()
        
        im = axes[0].imshow(
            freq.T, 
            origin='lower',
            aspect='auto',
            extent=[self.x_range[0], self.x_range[1], 
                    self.y_range[0], self.y_range[1]],
            cmap='hot'
        )
        
        axes[0].set_xlabel(self.x_label)
        axes[0].set_ylabel(self.y_label)
        axes[0].set_title(f'State Visitation (Coverage: {self.get_coverage():.1%})')
        plt.colorbar(im, ax=axes[0], label='Visit Frequency')
        
        if include_actions and len(axes) > 1:
            if action_names is None:
                action_names = [f'Action {i}' for i in range(n_actions)]
            
            for i in range(n_actions):
                action_map = self.get_action_heatmap(i)
                
                im = axes[i + 1].imshow(
                    action_map.T,
                    origin='lower',
                    aspect='auto',
                    extent=[self.x_range[0], self.x_range[1],
                            self.y_range[0], self.y_range[1]],
                    cmap='viridis',
                    vmin=0, vmax=1
                )
                
                axes[i + 1].set_xlabel(self.x_label)
                axes[i + 1].set_ylabel(self.y_label)
                axes[i + 1].set_title(f'{action_names[i]} Probability')
                plt.colorbar(im, ax=axes[i + 1], label='P(action)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def reset(self) -> None:
                                     
        self._visit_counts = np.zeros(self.bins)
        self._state_actions.clear()
        self._total_visits = 0
