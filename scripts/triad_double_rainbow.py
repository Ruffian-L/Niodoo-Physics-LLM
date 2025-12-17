#!/usr/bin/env python3
"""
🌈🔥 TRIAD DOUBLE RAINBOW STRESS TEST V2
Testing all three cognitive stressors with telemetry output.
"""

import subprocess
import re
import sys
import os
import json

BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# High-energy blend range - finer granularity
BLENDS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# THE TRIAD
PROMPTS = [
    ("🦁 LOGIC", "If I put a coin in a cup and put the cup in the freezer, is the coin wet or dry? Explain step by step."),
    ("🎨 CREATIVE", "Write a noir detective opening line set on a rainy Mars colony."),
    ("🌫️ AMBIGUITY", "What is the best way to fix it?")
]

def run_test(blend, label, prompt):
    cmd = [
        BINARY, 
        "--model-path", MODEL, 
        "--prompt", prompt,
        "--mode-orbital",
        f"--physics-blend={blend}",
        "--repulsion-strength=-2.0",
        "--orbit-speed=0.15",
        "--gravity-well=1.0",
        "--max-steps", "120",  # More tokens
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        output = ""
        debris_count = 0
        telemetry = []
        
        for line in result.stdout.split('\n'):
            if "[DBG: Decoded" in line:
                output += line.split("'")[1].replace("\\n", "\n").replace("\\'", "'")
            if "[DEBRIS]" in line:
                debris_count += 1
                print(f"    {line.strip()}")  # Stream debris to console
            if "[TELEMETRY]" in line:
                try:
                    telem_json = line.split("[TELEMETRY]")[1].strip()
                    telemetry.append(json.loads(telem_json))
                except:
                    pass
                    
        return output.strip(), debris_count, telemetry
    except Exception as e:
        return f"[ERROR: {e}]", -1, []

if __name__ == "__main__":
    if not os.path.exists(BINARY):
        print("Build binary first!")
        sys.exit(1)

    print("=" * 80)
    print("🌈🔥 TRIAD DOUBLE RAINBOW V2 (DEBRIS FILTER + TELEMETRY)")
    print("=" * 80)
    
    with open("artifacts/triad_double_rainbow_v2.md", "w") as f:
        f.write("# 🌈🔥 Triad Double Rainbow Stress Test V2\n\n")
        f.write("**Debris Filter: ACTIVE (with Angeles/assistant blocked)**\n\n")
        f.write("Testing Logic/Creative/Ambiguity across Blend 1.0-5.0\n\n")
        
        for blend in BLENDS:
            print(f"\n{'='*80}")
            print(f"🔥 BLEND: {blend}")
            print(f"{'='*80}")
            
            f.write(f"\n## Blend {blend}\n")
            
            for label, prompt in PROMPTS:
                print(f"\n{label}:")
                print("-" * 40)
                
                text, debris, telemetry = run_test(blend, label, prompt)
                
                # Full text to console
                print(text)
                if debris > 0:
                    print(f"\n  [⚠️ DEBRIS BLOCKED: {debris} tokens]")
                
                # Telemetry summary
                if telemetry:
                    avg_force = sum(t.get('total_force', 0) for t in telemetry) / len(telemetry)
                    glitches = sum(1 for t in telemetry if t.get('is_glitch', False))
                    print(f"  [📊 AVG_FORCE: {avg_force:.4f}, GLITCHES: {glitches}]")
                
                # Save to artifact
                f.write(f"\n### {label}\n")
                if debris > 0:
                    f.write(f"**⚠️ DEBRIS BLOCKED: {debris} tokens**\n\n")
                f.write(f"```text\n{text}\n```\n\n")
                
                # Save telemetry
                if telemetry:
                    avg_force = sum(t.get('total_force', 0) for t in telemetry) / len(telemetry)
                    glitches = sum(1 for t in telemetry if t.get('is_glitch', False))
                    f.write(f"**Telemetry:** Avg Force={avg_force:.4f}, Glitches={glitches}\n\n")
                
                f.flush()
    
    print("\n" + "=" * 80)
    print("DONE. Results saved to artifacts/triad_double_rainbow_v2.md")
