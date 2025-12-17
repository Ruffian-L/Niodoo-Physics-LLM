#!/usr/bin/env python3
"""
🪐 NIODOO ORBITAL TUNER V2 (HIGH ENERGY)
Tuning Orbit Speed vs. Gravity with Unclamped Physics.
"""

import subprocess
import re
import sys
import os

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================
BINARY_PATH = "./target/release/niodoo"
MODEL_PATH = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# CONSTANTS (From the previous Rainbow Tune)
FIXED_BLEND = "1.0"
FIXED_REPULSION = "-1.0"

# X-Axis: ORBIT SPEED (How fast we drift)
# We expect the new ceiling to be lower than before.
SPEEDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# Y-Axis: GRAVITY WELL (The Anchor)
# How hard does the prompt pull back?
GRAVITIES = [0.5, 1.0, 2.0, 5.0]

# The "Canary in the Coal Mine" Prompt
TEST_PROMPT = "Write a noir detective opening line set on a rainy Mars colony."

# ==============================================================================
# MAIN
# ==============================================================================
def run_orbit_test(speed, grav):
    cmd = [
        BINARY_PATH,
        "--model-path", MODEL_PATH,
        "--prompt", TEST_PROMPT,
        "--mode-orbital",
        f"--orbit-speed={speed}",
        f"--gravity-well={grav}",
        f"--physics-blend={FIXED_BLEND}",
        f"--repulsion-strength={FIXED_REPULSION}",
        "--max-steps", "64",
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # Extract text
        output = ""
        token_pattern = re.compile(r"\[DBG: Decoded '(.*)'\]")
        for line in result.stdout.split('\n'):
            m = token_pattern.search(line)
            if m:
                output += m.group(1).replace("\\n", " ").replace("\\'", "'")
        
        return output.strip()
    except:
        return "[TIMEOUT]"

if __name__ == "__main__":
    if not os.path.exists(BINARY_PATH):
        print("Binary missing.")
        sys.exit(1)

    print("🪐 HIGH-ENERGY ORBITAL SWEEP")
    print(f"Fixed: Blend {FIXED_BLEND} | Repulsion {FIXED_REPULSION}")
    print("="*60)
    
    # Open log file
    with open("artifacts/orbital_v2_outputs.md", "w") as f:
        f.write("# 🪐 Orbital Tuner V2 (High Energy)\n\n")
        f.write(f"**Fixed:** Blend {FIXED_BLEND} | Repulsion {FIXED_REPULSION}\n\n")
        f.write(f"**Prompt:** \"{TEST_PROMPT}\"\n\n")
        
        for g in GRAVITIES:
            print(f"\n▬▬▬ GRAVITY: {g} ▬▬▬")
            f.write(f"\n## Gravity: {g}\n")
            
            for s in SPEEDS:
                print(f"\n🔸 Speed {s}:")
                f.write(f"\n### Speed {s}\n```text\n")
                
                text = run_orbit_test(s, g)
                print(text)
                
                f.write(text + "\n```\n")
                f.flush()
    
    print("\n" + "="*60)
    print("DONE. Outputs saved to artifacts/orbital_v2_outputs.md")
