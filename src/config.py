import yaml
import json
import hashlib
import os

def load_config(file_path: str, required_keys: list = None) -> tuple[dict, str]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    if required_keys:
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key in config {file_path}: {key}")
                
    # Calculate hash
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    
    # Save resolved config
    os.makedirs('outputs/metrics', exist_ok=True)
    base_name = os.path.basename(file_path).split('.')[0]
    
    resolved_path = f'outputs/metrics/resolved_config_{base_name}.json'
    with open(resolved_path, 'w', encoding='utf-8') as f:
        json.dump({"config": config, "config_hash": config_hash}, f, indent=2)
        
    return config, config_hash
