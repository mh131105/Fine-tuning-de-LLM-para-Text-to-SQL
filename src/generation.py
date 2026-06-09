import time
import json
import os

def generate_text(model, tokenizer, prompt: str, gen_config: dict) -> tuple[str, float]:
    start_time = time.time()
    
    # If no model (e.g. testing mode or mock mode), return dummy text
    if model is None:
        return "SELECT * FROM dummy;", 0.1
        
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=gen_config.get("max_new_tokens", 256),
        do_sample=gen_config.get("do_sample", False),
        temperature=gen_config.get("temperature", 0.0),
        top_p=gen_config.get("top_p", 1.0),
        num_beams=gen_config.get("num_beams", 1),
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode only the new tokens
    input_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0][input_length:]
    
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    latency = time.time() - start_time
    
    # Save config
    os.makedirs('outputs/metrics', exist_ok=True)
    with open('outputs/metrics/generation_config.json', 'w') as f:
        json.dump(gen_config, f, indent=2)
        
    return output_text, latency
