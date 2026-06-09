import os
import json
import platform
import sys

def record_environment():
    env_data = {
        "python_version": sys.version,
        "os": platform.system(),
        "gpu_name": None,
        "gpu_vram": None,
        "cuda_version": None,
        "torch_version": None
    }
    
    try:
        import torch
        env_data["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            env_data["cuda_version"] = torch.version.cuda
            env_data["gpu_name"] = torch.cuda.get_device_name(0)
            env_data["gpu_vram"] = torch.cuda.get_device_properties(0).total_memory / (1024**3) # in GB
    except ImportError:
        pass
        
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/environment.json", "w", encoding='utf-8') as f:
        json.dump(env_data, f, indent=2)
        
    return env_data

if __name__ == "__main__":
    print(json.dumps(record_environment(), indent=2))
