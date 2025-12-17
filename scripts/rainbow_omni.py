#!/usr/bin/env python3
"""
🌈 RAINBOW OMNI-TUNER V3 (Niodoo v3.0 Elastic)
Extended Token Limits (256) to verify drift-snap cycle.
Focus: Logic, Creative, Ambiguity (The Triad).
"""

import subprocess
import time
import os
import sys
import json
import itertools
from datetime import datetime

# --- CONFIGURATION ---
BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# The "Triad" Prompts
PROMPTS = {
    "Logic": "Is a tomato a fruit or a vegetable?",
    "Creative": "Describe the taste of starlight.",
    "Ambiguity": "I went to the bank."
}

# Rainbow Grid V3
GRID = {
    "physics_blend": [1.0],         # Locked High Energy
    "repulsion_strength": [-0.5, -1.0, -1.5],
    "orbit_speed": [0.1, 0.2],
    "gravity_well": [0.2, 0.5, 0.8]
}

# --- LOGGING ---
LOG_DIR = "artifacts/logs/rainbow_v3"
os.makedirs(LOG_DIR, exist_ok=True)
FULL_LOG_FILE = "artifacts/rainbow_omni_v3_log.md"
METRICS_FILE = "artifacts/rainbow_omni_v3_metrics.csv"

def parse_telemetry(line):
    if "[TELEMETRY]" in line:
        try:
            json_str = line.split("[TELEMETRY]")[1].strip()
            return json.loads(json_str)
        except:
            return None
    return None

def run_test(prompt_name, prompt_text, params):
    blend = params["physics_blend"]
    repulsion = params["repulsion_strength"]
    speed = params["orbit_speed"]
    gravity = params["gravity_well"]
    
    cmd = [
        BINARY, "--model-path", MODEL, "--prompt", prompt_text,
        "--mode-orbital", 
        f"--physics-blend={blend}", 
        f"--repulsion-strength={repulsion}",
        "--physics-start-layer=16", 
        "--physics-end-layer=31",
        f"--orbit-speed={speed}",
        f"--gravity-well={gravity}",
        "--max-steps", "256", "--seed", "42"
    ]
    
    print(f"\n🚀 {prompt_name} | B={blend} R={repulsion} S={speed} G={gravity}")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        full_log_content = f"## Run: {prompt_name} | Blend {blend} | Rep {repulsion} | Speed {speed} | Grav {gravity}\n"
        full_log_content += f"**Prompt:** {prompt_text}\n\n```\n"
        
        generated_text = ""
        telemetry_data = []
        glitch_count = 0
        elastic_logs = []
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                if "[DBG: Decoded" in line:
                    try:
                        token = line.split("'")[1].replace("\\n", "\n")
                        generated_text += token
                        print(token, end="", flush=True) # Stream text
                        full_log_content += token
                        
                        if any(x in token for x in ["://", ".Forms", "php"]):
                            glitch_count += 1
                    except: pass
                
                if "[TELEMETRY]" in line:
                    data = parse_telemetry(line)
                    if data: telemetry_data.append(data)

                if "[ELASTIC" in line:
                    elastic_logs.append(line.strip())
        
        full_log_content += "\n```\n\n**Elastic Telemetry Sample:**\n```\n"
        # Include a few elastic samples
        for log in elastic_logs[::5]: # Sample every 5th
             full_log_content += log + "\n"
        full_log_content += "```\n---\n\n"
        
        print("\n") # Newline after stream
        
        process.wait()
        
        # Calculate Metrics
        avg_force = 0
        if telemetry_data:
            avg_force = sum(d.get("total_force", 0) for d in telemetry_data) / len(telemetry_data)
        
        len_tokens = len(telemetry_data)
        
        return {
            "text": full_log_content,
            "metrics": [blend, repulsion, speed, gravity, prompt_name, f"{avg_force:.4f}", len_tokens, glitch_count]
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("🌈 RAINBOW OMNI-TUNER V3 STARTED")
    
    # 2. Prepare Output Files
    with open(METRICS_FILE, "w") as f:
        f.write("Blend,Repulsion,Speed,Gravity,Prompt,Total_Force,Length,Glitch_Count\n")
        
    with open(FULL_LOG_FILE, "w") as f:
        f.write("# 🌈 Rainbow Omni-Tuner V3: Full Log\n\n")

    # 3. Running Grid
    keys = GRID.keys()
    values = GRID.values()
    combinations = list(itertools.product(*values))
    
    total = len(combinations) * len(PROMPTS)
    current = 0
    
    for combo in combinations:
        params = dict(zip(keys, combo))
        
        for p_name, p_text in PROMPTS.items():
            current += 1
            print(f"--- Run {current}/{total} ---")
            
            result = run_test(p_name, p_text, params)
            
            if result:
                # Append Text Log
                with open(FULL_LOG_FILE, "a") as f:
                    f.write(result["text"])
                    
                # Append Metrics
                with open(METRICS_FILE, "a") as f:
                    line = ",".join(map(str, result["metrics"])) + "\n"
                    f.write(line)

    print(f"\n🏁 ALL DONE. \n- Logs: {FULL_LOG_FILE}\n- Metrics: {METRICS_FILE}")

if __name__ == "__main__":
    main()
