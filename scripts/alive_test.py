#!/usr/bin/env python3
"""
🧠 ALIVE TEST (Elastic Gravity V2.2)
Running "High Force" prompts to verify Cognitive Wobble.
Telemtry [ELASTIC] should be visible.
"""

import subprocess
import time
import os
import sys

# --- CONFIG ---
BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# The "Alive" Suite
PROMPTS = [
    "Write a poem about the moon.",
    "Tell me the truth.",
    "I saw the man with the telescope. Who had the telescope?",
    "What is consciousness?",
    "Why did you say that?"
]

LOG_FILE = "artifacts/alive_test_log.md"

def run_alive(prompt):
    cmd = [
        BINARY, "--model-path", MODEL, "--prompt", prompt,
        "--mode-orbital", # Defaults: Blend 1.2, Rep -1.3, Speed 0.2
        "--max-steps", "256", "--seed", "42"
    ]
    
    print(f"\n🚀 PROMPT: {prompt}")
    print(f"   (Running High Force: Blend 1.2, Rep -1.3, Speed 0.2)")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        
        full_output = f"## Prompt: {prompt}\n\n```\n"
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                # Print everything to console (for "Drop the log")
                print(line, end="")
                
                # Capture text and meaningful telemetry for file
                if "[ELASTIC]" in line or "[DBG: Decoded" in line:
                    full_output += line
                
                # If decoded token, strip logging prefix for cleaner reading in markdown?
                # Actually, user wants "Drop the log", so raw is good.
                # But for the artifact, maybe readable?
                if "[DBG: Decoded" in line:
                     try:
                        token = line.split("'")[1].replace("\\n", "\n")
                        # full_output += token # Maybe duplicate text separately?
                     except: pass

        full_output += "\n```\n\n---\n"
        return full_output
        
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}\n"

def main():
    print("🧠 ALIVE TEST STARTED")
    
    with open(LOG_FILE, "w") as f:
        f.write("# 🧠 Alive Test Log (Elastic Gravity V2.2)\n\n")

    for prompt in PROMPTS:
        result = run_alive(prompt)
        with open(LOG_FILE, "a") as f:
            f.write(result)
            
    print(f"\n✅ DONE. Log: {LOG_FILE}")

if __name__ == "__main__":
    main()
