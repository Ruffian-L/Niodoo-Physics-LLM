#!/usr/bin/env python3
"""
🧠 HARD LOGIC V2: VANILLA FAILURE CHECK + NIODOO REDEMPTION
Target: Verify Vanilla Llama failure on 5 specific prompts, then test Niodoo Recovery.
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

PROMPTS = {
    "1_BatBall": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning step by step.",
    "2_LilyPad": "A lily pad doubles in size every day. On day 30, it covers the entire pond. On what day did it cover half the pond? Explain step by step.",
    "3_MissingDollar": "Three guests check into a hotel room that costs $30. They each pay $10. Later, the manager realizes the room is $25, so he sends the bellboy with $5 refund. The bellboy pockets $2 and gives $3 back. Each guest paid $9 ($27 total), bellboy has $2 — $29. Where's the missing $1? Explain step by step.",
    "4_LiarParadox": "The sentence below is false.\nThe sentence above is true.\nWhich sentence is true, which is false? Or is there a paradox? Explain carefully step by step.",
    "5_MontyHall100": "You’re on a game show with 100 doors. Behind 1 door is a car, behind 99 are goats. You pick door 1. Host opens 98 doors with goats (leaves door 1 and door 100 closed). Do you switch to door 100? Why? Calculate probabilities step by step."
}

OUTPUT_FILE = "artifacts/hard_logic_v2_comparison.md"

def run_vanilla(prompt):
    print(f"   🍦 Running Vanilla ({OLLAMA_MODEL})...")
    try:
        # Default Ollama run (Temperature is generally 0.8)
        cmd = ["ollama", "run", OLLAMA_MODEL, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error running vanilla: {e}"

def run_niodoo(prompt):
    print(f"   🧠 Running Niodoo v3 (Gravity 0.5 - High Force)...")
    
    # Niodoo v3 settings (High Force)
    cmd = (
        f"{BINARY} --model-path {MODEL} "
        f"--prompt {shlex.quote(prompt)} "
        "--mode-orbital "
        "--physics-blend 1.2 "
        "--repulsion-strength=-1.3 "
        "--orbit-speed 0.2 "
        "--gravity-well 0.5 " 
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
    print("🧠 HARD LOGIC V2 STARTED")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("# 🧠 Hard Logic V2: Vanilla Failure vs Niodoo Redemption\n\n")

    for name, prompt in PROMPTS.items():
        print(f"\n--- TEST: {name} ---")
        
        # 1. Vanilla
        print(f"Prompt: {prompt[:50]}...")
        vanilla = run_vanilla(prompt)
        
        # 2. Niodoo
        niodoo, telemetry = run_niodoo(prompt)

        # Save Result
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"## {name}\n")
            f.write(f"**Prompt:** {prompt}\n\n")
            f.write("### 🍦 Vanilla Response\n")
            f.write(f"{vanilla}\n\n")
            f.write("### 🧠 Niodoo v3 Response\n")
            f.write(f"{niodoo}\n\n")
            f.write("### Telemetry\n```\n")
            lines = telemetry.strip().split('\n')
            if len(lines) > 6: 
                 f.write("\n".join(lines[:3]) + "\n...\n" + "\n".join(lines[-3:]))
            else:
                 f.write(telemetry)
            f.write("\n```\n\n---\n")

    print(f"\n✅ DONE. Check {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
