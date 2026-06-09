import random
import numpy as np
import json
import os

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Make deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
        
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/reproducibility.json", "w", encoding='utf-8') as f:
        json.dump({"seed": seed}, f, indent=2)
