#!/usr/bin/env python3
"""
🧠 HARD LOGIC STRESS TEST (Vanilla vs Niodoo v3 Elastic)
Running 6 "Hard" Reasoning Prompts to verify "Wobble & Recovery".

Configuration:
- Vanilla: Ollama run (Llama-3.1-8B-Instruct)
- Niodoo: High Force (Blend 1.2, Rep -1.3, Speed 0.2, Gravity 0.5->Elastic)
"""

import subprocess
import time
import os
import sys

# --- CONFIG ---
BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OLLAMA_MODEL = "llama31-local:latest" # Explicit tag

PROMPTS = {
    "1_BatBall": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain your reasoning step by step.",
    "2_LilyPad": "A lily pad doubles in size every day. On day 30, it covers the entire pond. On what day did it cover half the pond? Explain step by step.",
    "3_MissingDollar": "Three people check into a hotel room that costs $30. They each pay $10. Later, the manager realizes the room is only $25, so he sends the bellboy with $5 to refund them. The bellboy pockets $2 and gives them $3 back. So each guest paid $9 ($27 total) and the bellboy has $2 — that's $29. Where is the missing $1? Explain step by step.",
    "4_Paradox": "This sentence is false. Is the above sentence true or false? Explain your reasoning carefully, step by step.",
    "5_Prisoners": "There are 100 prisoners. Each wears a red or blue hat, randomly assigned. They can see everyone else's hat but not their own. They must guess their own hat color simultaneously. If at least one guesses correctly, they all go free. They can plan beforehand. What strategy guarantees at least one correct guess? Explain step by step.",
    "6_Telescope": "I saw the man with the telescope at the bank. Who had the telescope? Who went to the bank? Explain all possible meanings."
}

OUTPUT_FILE = "artifacts/hard_logic_comparison.md"

def run_vanilla(prompt):
    print(f"   🍦 Running Vanilla...")
    try:
        # Try running via ollama
        cmd = ["ollama", "run", "llama3.1", prompt]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception as e:
        return f"Error running vanilla: {e}"

def run_niodoo(prompt):
    print(f"   🧠 Running Niodoo v3 (High Force)...")
    import shlex
    
    # Construct shell command carefully
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
        return f"Error: {e}", ""

def main():
    print("🧠 HARD LOGIC COMPARED: Vanilla vs Niodoo v3")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("# 🧠 Hard Logic Comparison: Vanilla vs Niodoo v3\n")
        f.write("**Settings:** High Force (Blend 1.2, Rep -1.3, Elastic Gravity)\n\n")
    
    for name, prompt in PROMPTS.items():
        print(f"\n--- TEST: {name} ---")
        
        # 1. Vanilla
        vanilla_response = run_vanilla(prompt)
        
        # 2. Niodoo
        niodoo_response, telemetry = run_niodoo(prompt)
        
        # Save
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"## {name}\n")
            f.write(f"**Prompt:** {prompt}\n\n")
            f.write("### 🍦 Vanilla Response\n")
            f.write(f"{vanilla_response}\n\n")
            f.write("### 🧠 Niodoo v3 Response\n")
            f.write(f"{niodoo_response}\n\n")
            f.write("**Elastic Telemetry Sample:**\n")
            f.write("```\n")
            # Sample first few and last few lines of telemetry
            lines = telemetry.strip().split('\n')
            if len(lines) > 10:
                f.write("\n".join(lines[:5]) + "\n...\n" + "\n".join(lines[-5:]))
            else:
                f.write(telemetry)
            f.write("\n```\n")
            f.write("\n---\n")

    print(f"\n✅ DONE. Comparison saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
