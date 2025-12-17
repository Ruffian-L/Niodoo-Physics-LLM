#!/usr/bin/env python3
"""
⚠️ NIODOO TRIAD TUNER
Micro-tuning Stability across Logic, Creativity, and Ambiguity.
"""

import subprocess
import sys
import os

BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# === THE MICRO-TUNE GRID ===
# We know 0.5 is weak. We know 3.0 is crazy.
# We search for the decimal perfect point.
BLENDS = [1.0, 1.25, 1.5, 1.75, 2.0]
REPULSIONS = [-1.0, -2.0, -3.0]

# === THE TRIAD ===
PROMPTS = [
    ("🦁 LOGIC", "If I put a coin in a cup and put the cup in the freezer, is the coin wet or dry? Explain step by step."),
    ("🎨 CREATIVE", "Write a noir detective opening line set on a rainy Mars colony."),
    ("🌫️ AMBIGUITY", "What is the best way to fix it?")
]

def run_test(blend, rep, label, prompt):
    print(f"   Running {label}...", end="\r")
    cmd = [
        BINARY, 
        "--model-path", MODEL, 
        "--prompt", prompt,
        "--mode-orbital",
        "--orbit-speed=0.15",      # Locked based on previous success
        "--gravity-well=1.0",      # Locked
        f"--physics-blend={blend}",
        f"--repulsion-strength={rep}",
        "--max-steps", "64",
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        output = ""
        for line in result.stdout.split('\n'):
            if "[DBG: Decoded" in line:
                output += line.split("'")[1].replace("\\n", " ")
        return output.strip()
    except:
        return "TIMEOUT/CRASH"

if __name__ == "__main__":
    if not os.path.exists(BINARY):
        print("Build binary first!")
        sys.exit(1)

    print("=== 🔬 NIODOO TRIAD MICRO-TUNE ===")
    
    # Open log file
    with open("artifacts/triad_tune_outputs.md", "w") as f:
        f.write("# 🔬 Niodoo Triad Micro-Tune Results\n\n")
        
        for rep in REPULSIONS:
            header = f"\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬\n🛡️  REPULSION: {rep}\n▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬"
            print(header)
            f.write(f"\n## Repulsion: {rep}\n")
            
            for blend in BLENDS:
                blend_header = f"\n🔸 BLEND: {blend}\n   {'-'*40}"
                print(blend_header)
                f.write(f"\n### Blend {blend}\n")
                
                for label, prompt in PROMPTS:
                    out = run_test(blend, rep, label, prompt)
                    
                    # Print full output to console
                    print(f"\n   {label}:")
                    print(f"   {out}")
                    
                    # Save full output to file
                    f.write(f"\n**{label}**\n```\n{out}\n```\n")
                    f.flush()
    
    print("\n" + "="*60)
    print("DONE. Full outputs saved to artifacts/triad_tune_outputs.md")
