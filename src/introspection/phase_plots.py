   
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

@dataclass
class Trajectory:
                                                 
    states: np.ndarray                  
    actions: np.ndarray        
    rewards: np.ndarray        
    episode: int
    
    @property
    def length(self) -> int:
        return len(self.states)
    
    @property
    def total_reward(self) -> float:
        return np.sum(self.rewards)

class PhasePlotter:
           
    def __init__(
        self,
        x_dim: int = 0,
        y_dim: int = 1,
        x_label: str = "Salary (normalized)",
        y_label: str = "Burnout"
    ):
                   
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.x_label = x_label
        self.y_label = y_label
        
        self._trajectories: List[Trajectory] = []
    
    def add_trajectory(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        episode: int
    ) -> None:
                                                 
        traj = Trajectory(
            states=np.array(states),
            actions=np.array(actions),
            rewards=np.array(rewards),
            episode=episode
        )
        self._trajectories.append(traj)
    
    def plot_single(
        self,
        trajectory: Trajectory,
        ax: Optional[plt.Axes] = None,
        color_by: str = "time",
        show_actions: bool = True,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
                   
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            standalone = True
        else:
            standalone = False
        
        x = trajectory.states[:, self.x_dim]
        y = trajectory.states[:, self.y_dim]
        
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        if color_by == "time":
            colors = np.linspace(0, 1, len(segments))
            cmap = "viridis"
            label = "Time (early → late)"
        elif color_by == "reward":
            colors = trajectory.rewards[:-1]
            cmap = "RdYlGn"
            label = "Reward"
        elif color_by == "action":
            colors = trajectory.actions[:-1].astype(float)
            cmap = "coolwarm"
            label = "Action (STAY=0, SWITCH=1)"
        else:
            colors = np.ones(len(segments))
            cmap = "Blues"
            label = None
        
        lc = LineCollection(segments, cmap=cmap, array=colors)
        lc.set_linewidth(2)
        line = ax.add_collection(lc)
        
        if label:
            plt.colorbar(line, ax=ax, label=label)
        
        ax.scatter([x[0]], [y[0]], c='green', s=100, marker='o', 
                  label='Start', zorder=5)
        ax.scatter([x[-1]], [y[-1]], c='red', s=100, marker='X', 
                  label='End', zorder=5)
        
        if show_actions:
            switch_mask = trajectory.actions == 1
            if np.any(switch_mask):
                switch_idx = np.where(switch_mask)[0]
                ax.scatter(x[switch_idx], y[switch_idx], c='orange', s=50, 
                          marker='^', label='SWITCH', alpha=0.7, zorder=4)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(f'Episode {trajectory.episode} (Return: {trajectory.total_reward:.2f})')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        if standalone:
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if show:
                plt.show()
            plt.close()
    
    def plot_multiple(
        self,
        n_trajectories: Optional[int] = None,
        selection: str = "last",
        alpha: float = 0.5,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
                   
        if not self._trajectories:
            print("No trajectories to plot")
            return
        
        trajs = self._select_trajectories(n_trajectories, selection)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        returns = [t.total_reward for t in trajs]
        min_ret, max_ret = min(returns), max(returns)
        
        for traj in trajs:
            x = traj.states[:, self.x_dim]
            y = traj.states[:, self.y_dim]
            
            if max_ret > min_ret:
                color_val = (traj.total_reward - min_ret) / (max_ret - min_ret)
            else:
                color_val = 0.5
            
            color = plt.cm.RdYlGn(color_val)
            ax.plot(x, y, color=color, alpha=alpha, linewidth=1)
            
            ax.scatter([x[-1]], [y[-1]], c=[color], s=30, marker='x', alpha=alpha)
        
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', 
                                    norm=plt.Normalize(min_ret, max_ret))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Episode Return')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(f'Phase Plot ({len(trajs)} trajectories, {selection})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def plot_evolution(
        self,
        episode_groups: List[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        show: bool = True
    ) -> None:
                   
        if not self._trajectories:
            print("No trajectories to plot")
            return
        
        if episode_groups is None:
                                     
            episodes = [t.episode for t in self._trajectories]
            min_ep, max_ep = min(episodes), max(episodes)
            step = (max_ep - min_ep) // 4
            episode_groups = [
                (min_ep, min_ep + step),
                (min_ep + step, min_ep + 2*step),
                (min_ep + 2*step, min_ep + 3*step),
                (min_ep + 3*step, max_ep + 1),
            ]
        
        fig, axes = plt.subplots(1, len(episode_groups), figsize=(5*len(episode_groups), 4))
        if len(episode_groups) == 1:
            axes = [axes]
        
        for ax, (start, end) in zip(axes, episode_groups):
            trajs = [t for t in self._trajectories if start <= t.episode < end]
            
            for traj in trajs:
                x = traj.states[:, self.x_dim]
                y = traj.states[:, self.y_dim]
                ax.plot(x, y, alpha=0.3, linewidth=0.5)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(self.x_label)
            ax.set_ylabel(self.y_label)
            ax.set_title(f'Episodes {start}-{end-1}\n({len(trajs)} trajectories)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def _select_trajectories(
        self,
        n: Optional[int],
        selection: str
    ) -> List[Trajectory]:
                                                    
        if n is None:
            n = len(self._trajectories)
        
        n = min(n, len(self._trajectories))
        
        if selection == "last":
            return self._trajectories[-n:]
        elif selection == "first":
            return self._trajectories[:n]
        elif selection == "random":
            import random
            return random.sample(self._trajectories, n)
        elif selection == "best":
            sorted_trajs = sorted(self._trajectories, 
                                 key=lambda t: t.total_reward, reverse=True)
            return sorted_trajs[:n]
        elif selection == "worst":
            sorted_trajs = sorted(self._trajectories, 
                                 key=lambda t: t.total_reward)
            return sorted_trajs[:n]
        else:
            return self._trajectories[-n:]
    
    def reset(self) -> None:
                                     
        self._trajectories.clear()
