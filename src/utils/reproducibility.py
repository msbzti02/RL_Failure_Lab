   
import os
import random
import hashlib
import platform
from datetime import datetime
from typing import Optional

import numpy as np
import torch

def set_global_seed(seed: int, deterministic: bool = True) -> None:
           
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
                                    
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                                                                          
                pass
    
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_reproducibility_info() -> dict:
           
    info = {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    
    return info

def compute_config_hash(config_dict: dict) -> str:
           
    import json
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

class SeedManager:
           
    def __init__(self, base_seed: int):
        self.base_seed = base_seed
        self._rng = np.random.RandomState(base_seed)
        self._seed_cache = {}
    
    def get_seed(self, component: str) -> int:
                                                             
        if component not in self._seed_cache:
                                                                           
            component_hash = int(hashlib.md5(component.encode()).hexdigest()[:8], 16)
            self._seed_cache[component] = (self.base_seed + component_hash) % (2**31)
        return self._seed_cache[component]
    
    def get_run_seed(self, run_index: int) -> int:
                                                                    
        return (self.base_seed + run_index * 1000) % (2**31)

if __name__ == "__main__":
          
    set_global_seed(42)
    print("Reproducibility info:")
    for k, v in get_reproducibility_info().items():
        print(f"  {k}: {v}")
    
    sm = SeedManager(42)
    print(f"\nEnv seed: {sm.get_seed('environment')}")
    print(f"Agent seed: {sm.get_seed('agent')}")
    print(f"Run 0 seed: {sm.get_run_seed(0)}")
    print(f"Run 1 seed: {sm.get_run_seed(1)}")
