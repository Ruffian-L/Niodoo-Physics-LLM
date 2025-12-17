#!/usr/bin/env python3
"""
🍓 THE STRAWBERRY GRAVITY SWEEP
Target: Find the optimal Elastic Gravity to count 'r's in "Strawberry".
Sweeping Gravity: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
"""

import subprocess
import time
import os
import sys
import shlex

# --- CONFIG ---
BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OLLAMA_MODEL = "llama31-local:latest"

PROMPT = "How many times does the letter 'r' appear in the word 'Strawberry'? Think through it step by step."
OUTPUT_FILE = "artifacts/strawberry_sweep_result.md"

GRAVITIES = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

def run_vanilla(prompt):
    print(f"   🍦 Running Vanilla ({OLLAMA_MODEL})...")
    try:
        cmd = ["ollama", "run", OLLAMA_MODEL, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error running vanilla: {e}"

def run_niodoo(prompt, gravity):
    print(f"   🧠 Running Niodoo v3 (Gravity {gravity})...")
    
    cmd = (
        f"{BINARY} --model-path {MODEL} "
        f"--prompt {shlex.quote(prompt)} "
        "--mode-orbital "
        "--physics-blend 1.2 "
        "--repulsion-strength=-1.3 "
        "--orbit-speed 0.2 "
        f"--gravity-well {gravity} " 
        "--max-steps 512 --seed 42"
    )
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, shell=True)
        full_output = ""
        elastic_telemetry = ""
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                if "[DBG: Decoded" in line:
                    try:
                        token = line.split("'")[1].replace("\\n", "\n")
                        full_output += token
                        print(token, end="", flush=True)
                    except: pass
                
                if "[ELASTIC" in line:
                    elastic_telemetry += line
                    
        print("\n")
        return full_output, elastic_telemetry
        
    except Exception as e:
        print(f"Error Niodoo: {e}")
        return f"Error: {e}", ""

def main():
    print("🍓 STRAWBERRY GRAVITY SWEEP STARTED")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("# 🍓 Strawberry Gravity Sweep\n\n")
        f.write(f"**Prompt:** {PROMPT}\n")
        f.write("**Goal:** Find the Gravity that breaks token blindness.\n\n")

    # 1. Vanilla Baseline
    vanilla = run_vanilla(PROMPT)
    with open(OUTPUT_FILE, "a") as f:
        f.write("## 🍦 Vanilla Baseline\n")
        f.write(f"{vanilla}\n\n---\n")

    # 2. Gravity Sweep
    for g in GRAVITIES:
        print(f"\n--- Testing Gravity {g} ---")
        niodoo, telemetry = run_niodoo(PROMPT, g)
        
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"## 🧠 Niodoo (Gravity {g})\n")
            f.write(f"{niodoo}\n\n")
            f.write("### Telemetry Sample\n```\n")
            lines = telemetry.strip().split('\n')
            if len(lines) > 10: 
                 f.write("\n".join(lines[:5]) + "\n...\n" + "\n".join(lines[-5:]))
            else:
                 f.write(telemetry)
            f.write("\n```\n\n---\n")

    print(f"\n✅ DONE. Check {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
