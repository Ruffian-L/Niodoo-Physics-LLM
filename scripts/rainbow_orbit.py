#!/usr/bin/env python3
"""
🪐 NIODOO ORBITAL TUNER
Sweeps Orbit Speed vs. Gravity to find Stable Trajectories.
"""

import subprocess
import re
import sys
import os

# ==============================================================================
# 🎛️ CONFIGURATION GRID
# ==============================================================================
BINARY_PATH = "./target/release/niodoo"
MODEL_PATH = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# X-Axis: How fast we move (Tangential Velocity)
SPEEDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

# Y-Axis: How hard we pull back (Gravity Well Strength)
GRAVITIES = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0]

# Prompt: Needs to be open-ended enough to allow orbit, but specific enough to detect drift.
TEST_PROMPT = "Write a short poem about the moon."

class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def analyze_orbit(text):
    """
    Scoring Logic for Orbits:
    - 💀 CRASH: Short output, garbage.
    - 💤 STATIC: High repetition of words (Orbit decay).
    - 🛸 DRIFT: "Wednesday", "Crisis", "Politics" (Escaped gravity).
    - ✅ STABLE: Unique words, on-topic.
    """
    text_lower = text.lower()
    
    # 1. Crash Check
    if len(text) < 50 or "#ab" in text:
        return "💀 CRASH", Color.FAIL
        
    # 2. Drift Check (Hallucination detector for Moon prompt)
    # If it talks about things completely unrelated to night/sky
    drift_words = ["wednesday", "senate", "company", "budget", "crisis"]
    if any(w in text_lower for w in drift_words):
        return "🛸 DRIFT", Color.BLUE

    # 3. Static Check (Boredom)
    # If "moon" or "light" appears too many times
    if text_lower.count("moon") > 3 or text_lower.count("light") > 3:
        return "💤 STATIC", Color.HEADER

    # 4. Success
    return "✅ STABLE", Color.GREEN

def run_orbit_test(speed, gravity):
    cmd = [
        BINARY_PATH,
        "--model-path", MODEL_PATH,
        "--prompt", TEST_PROMPT,
        "--mode-orbital",
        f"--orbit-speed={speed}",
        f"--gravity-well={gravity}",
        "--physics-blend", "0.55",
        "--max-steps", "100",
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        
        # Clean output
        clean_text = ""
        token_pattern = re.compile(r"\[DBG: Decoded '(.*)'\]")
        for line in result.stdout.split('\n'):
            m = token_pattern.search(line)
            if m:
                clean_text += m.group(1).replace("\\n", " ").replace("\\'", "'")
                
        status, color = analyze_orbit(clean_text)
        return status, color, clean_text

    except:
        return "TIMEOUT", Color.FAIL, "TIMEOUT"

def print_table(results):
    with open("artifacts/rainbow_sweep_raw.log", "w") as f:
        f.write("🪐 ORBITAL STABILITY MATRIX (Gravity vs. Speed)\n")
        f.write("-" * 60 + "\n")
        headers = "".join([f"{s:>10}" for s in SPEEDS])
        f.write(f"{'GRAV \\ SPD':<12} | {headers}\n")
        f.write("-" * 60 + "\n")
        
        print(f"\n{Color.HEADER}🪐 ORBITAL STABILITY MATRIX (Gravity vs. Speed){Color.ENDC}")
        print("-" * 60)
        print(f"{'GRAV \\ SPD':<12} | {headers}")
        print("-" * 60)
        
        for g in GRAVITIES:
            row_str = f"{g:<12} | "
            file_row_str = f"{g:<12} | "
            for s in SPEEDS:
                if (s, g) in results:
                    status, color, _ = results[(s, g)]
                else:
                    status, color = "N/A", Color.ENDC
                # For console
                icon = "✅" if "STABLE" in status else "🛸" if "DRIFT" in status else "💤" if "STATIC" in status else "💀"
                row_str += f"{color}{icon:>10}{Color.ENDC}"
                
                # For log file (no color codes)
                text_icon = "[STABLE]" if "STABLE" in status else "[DRIFT]" if "DRIFT" in status else "[STATIC]" if "STATIC" in status else "[CRASH]"
                file_row_str += f"{text_icon:>10}"
            
            print(row_str)
            f.write(file_row_str + "\n")
            
    print(f"\n{Color.GREEN}Stability matrix saved to artifacts/rainbow_sweep_raw.log{Color.ENDC}")

def save_detailed_outputs(results):
    with open("artifacts/rainbow_outputs.md", "w") as f:
        f.write("# 🪐 Rainbow Orbit: Full Text Outputs\n\n")
        f.write("Prompt: \"Write a short poem about the moon.\"\n\n")
        
        for g in GRAVITIES:
            f.write(f"## Gravity Well: {g}\n")
            for s in SPEEDS:
                status, _, text = results.get((s, g), ("N/A", "", ""))
                f.write(f"### Speed {s} | {status}\n")
                f.write("```text\n")
                f.write(text.strip() + "\n")
                f.write("```\n\n")
    print(f"{Color.GREEN}Full text outputs saved to artifacts/rainbow_outputs.md{Color.ENDC}")

if __name__ == "__main__":
    if not os.path.exists(BINARY_PATH):
        print("Binary not found. Build it first!")
        sys.exit(1)

    results = {}
    print("Starting Orbital Sweep (Gravity vs Speed)...")
    
    for g in GRAVITIES:
        for speed in SPEEDS:
            print(f"Orbiting: Speed {speed} | Gravity {g}...", end="\r")
            print(f"Orbiting: Speed {speed} | Gravity {g}...", end="\r")
            status, color, text = run_orbit_test(speed, g)
            results[(speed, g)] = (status, color, text)
            
            if "STABLE" in status:
                 print(f"Stable Orbit found! Speed {speed} / Grav {g}      ")
                 
    print_table(results)
    save_detailed_outputs(results)
