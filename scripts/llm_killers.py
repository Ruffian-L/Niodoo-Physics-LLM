#!/usr/bin/env python3
"""
💀 LLM KILLERS: Semantic & Physical Common Sense Traps
These exploit LLMs predicting likely completion, not physical truth.
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

# The 3 LLM Killers - NO "step by step" to trigger intuitive failure
PROMPTS = {
    "1_KillersInRoom": "There are 3 killers in a room. I enter the room and kill one of them. How many killers are in the room now?",
    "2_TroyWeight": "Which weighs more: a pound of lead or a pound of gold?",
    "3_DryingTowels": "It takes 1 hour to dry one towel on a sunny clothesline. How long does it take to dry 50 towels?"
}

EXPECTED = {
    "1_KillersInRoom": "3 (or 4 counting body). You become a killer.",
    "2_TroyWeight": "Lead (Avoirdupois pound > Troy pound)",
    "3_DryingTowels": "1 hour (parallel drying)"
}

OUTPUT_FILE = "artifacts/llm_killers_result.md"

def run_vanilla(prompt):
    print(f"   🍦 Running Vanilla...")
    try:
        cmd = ["ollama", "run", OLLAMA_MODEL, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error running vanilla: {e}"

def run_niodoo(prompt):
    print(f"   🧠 Running Niodoo v3 (Gravity 0.6)...")
    
    cmd = (
        f"{BINARY} --model-path {MODEL} "
        f"--prompt {shlex.quote(prompt)} "
        "--mode-orbital "
        "--physics-blend 1.2 "
        "--repulsion-strength=-1.3 "
        "--orbit-speed 0.2 "
        "--gravity-well 0.6 " 
        "--max-steps 256 --seed 42"
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
    print("💀 LLM KILLERS TEST STARTED")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("# 💀 LLM Killers: Semantic Trap Results\n\n")

    for name, prompt in PROMPTS.items():
        print(f"\n--- TEST: {name} ---")
        print(f"Prompt: {prompt}")
        
        # 1. Vanilla
        vanilla = run_vanilla(prompt)
        
        # 2. Niodoo
        niodoo, telemetry = run_niodoo(prompt)

        # Save Result
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"## {name}\n")
            f.write(f"**Prompt:** {prompt}\n")
            f.write(f"**Expected Correct:** {EXPECTED[name]}\n\n")
            f.write("### 🍦 Vanilla Response\n")
            f.write(f"{vanilla}\n\n")
            f.write("### 🧠 Niodoo v3 Response\n")
            f.write(f"{niodoo}\n\n")
            f.write("---\n")

    print(f"\n✅ DONE. Check {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
