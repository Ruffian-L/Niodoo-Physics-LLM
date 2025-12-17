#!/usr/bin/env python3
import subprocess
import sys
import re

BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
PROMPT = "Write a short poem about the moon."

# The 3 Configurations to test
CONFIGS = [
    # 1. THE ROCK (Should be boring/repetitive)
    {"name": "STATIC (Low Speed, High Grav)", "speed": 0.05, "grav": 5.0},
    
    # 2. THE GOLD (Should be the target)
    {"name": "TARGET (Mid Speed, Mid Grav)",  "speed": 0.15, "grav": 1.0},
    
    # 3. THE COMET (Should be insane/hallucinating)
    {"name": "CHAOS (High Speed, Low Grav)",  "speed": 0.50, "grav": 0.1},
]

def run_test(config):
    print(f"\n🧪 TESTING: {config['name']}")
    print(f"   (Speed: {config['speed']} | Gravity: {config['grav']})")
    
    cmd = [
        BINARY, 
        "--model-path", MODEL, 
        "--prompt", PROMPT,
        "--mode-orbital",
        f"--orbit-speed={config['speed']}",
        f"--gravity-well={config['grav']}",
        "--max-steps", "64",
        "--seed", "42" # CONSTANT SEED to prove physics changes output
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Strip logs to show just the text
        output = ""
        # Improved regex to catch decoding log
        token_pattern = re.compile(r"\[DBG: Decoded '(.*)'\]")
        lines = result.stdout.split('\n')
        for line in lines:
            m = token_pattern.search(line)
            if m:
                clean = m.group(1).replace("\\n", "\n").replace("\\'", "'")
                output += clean
        
        # If regex failing (debug output format changed?), try robust fallback or print raw
        if not output.strip():
             # Fallback: Print lines that look like generation if any, or warn
             # Assuming standard stdout capture if DBG tags missing
             pass

        print(f"📝 OUTPUT:\n{'-'*40}")
        print(output.strip())
        print(f"{'-'*40}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    for conf in CONFIGS:
        run_test(conf)
