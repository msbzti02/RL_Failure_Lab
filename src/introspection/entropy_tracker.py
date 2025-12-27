   
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from collections import deque

class EntropyTracker:
           
    def __init__(self, window_size: int = 100):
                   
        self.window_size = window_size
        
        self._entropies: List[float] = []
        self._episodes: List[int] = []
        self._rolling_window = deque(maxlen=window_size)
    
    def update(self, entropy: float, episode: int) -> None:
                                   
        self._entropies.append(entropy)
        self._episodes.append(episode)
        self._rolling_window.append(entropy)
    
    def get_current(self) -> float:
                                      
        return self._entropies[-1] if self._entropies else 0.0
    
    def get_rolling_mean(self) -> float:
                                       
        return np.mean(self._rolling_window) if self._rolling_window else 0.0
    
    def get_rolling_std(self) -> float:
                                         
        return np.std(self._rolling_window) if len(self._rolling_window) > 1 else 0.0
    
    def get_trend(self) -> str:
                                    
        if len(self._entropies) < 20:
            return "insufficient_data"
        
        early = np.mean(self._entropies[:len(self._entropies)//4])
        late = np.mean(self._entropies[-len(self._entropies)//4:])
        
        if late < 0.1:
            if early > 0.5:
                return "collapsed"                               
            return "always_low"
        elif late > early * 1.5:
            return "increasing"                             
        elif late < early * 0.5:
            return "decreasing"                           
        else:
            return "stable"
    
    def get_diagnostics(self) -> Dict[str, float]:
                                     
        if not self._entropies:
            return {}
        
        return {
            "current": self._entropies[-1],
            "mean": np.mean(self._entropies),
            "min": np.min(self._entropies),
            "max": np.max(self._entropies),
            "std": np.std(self._entropies),
            "rolling_mean": self.get_rolling_mean(),
            "trend": self.get_trend(),
        }
    
    def plot(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        threshold: float = 0.1
    ) -> None:
                   
        if not self._entropies:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        ax1.plot(self._episodes, self._entropies, alpha=0.7, label='Entropy')
        
        if len(self._entropies) >= self.window_size:
            rolling = np.convolve(self._entropies, 
                                 np.ones(self.window_size)/self.window_size, 
                                 mode='valid')
            episodes_rolling = self._episodes[self.window_size-1:]
            ax1.plot(episodes_rolling, rolling, 'r-', lw=2, label='Rolling Mean')
        
        ax1.axhline(y=threshold, color='orange', linestyle='--', 
                   label=f'Collapse Threshold ({threshold})')
        
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Policy Entropy')
        ax1.set_title('Policy Entropy Over Training')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.hist(self._entropies, bins=30, density=True, alpha=0.7, color='blue')
        ax2.axvline(x=threshold, color='orange', linestyle='--',
                   label=f'Collapse Threshold')
        ax2.axvline(x=np.mean(self._entropies), color='red', linestyle='-',
                   label=f'Mean: {np.mean(self._entropies):.3f}')
        
        ax2.set_xlabel('Entropy')
        ax2.set_ylabel('Density')
        ax2.set_title('Entropy Distribution')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    def reset(self) -> None:
                                     
        self._entropies.clear()
        self._episodes.clear()
        self._rolling_window.clear()
