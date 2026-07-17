#!/usr/bin/env python3
"""
Baseline vs Golden (reference config) comparison
Shows the difference between vanilla Llama-3 and Niodoo Physics.
"""

import subprocess
import re
import sys
import os

BINARY = "./target/release/niodoo"
MODEL = os.environ.get("NIODOO_MODEL", "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

# PROMPT CATEGORIES (Logic / Creative / Ambiguity)
PROMPTS = [
    ("🦁 LOGIC", "If I put a coin in a cup and put the cup in the freezer, is the coin wet or dry? Explain step by step."),
    ("🎨 CREATIVE", "Write a noir detective opening line set on a rainy Mars colony."),
    ("🌫️ AMBIGUITY", "What is the best way to fix it?")
]

def run_test(prompt, use_physics=False):
    cmd = [
        BINARY, 
        "--model-path", MODEL, 
        "--prompt", prompt,
        "--max-steps", "80",
        "--seed", "42"
    ]
    
    if use_physics:
        cmd.extend([
            "--mode-orbital",
            "--physics-blend=1.0",
            "--repulsion-strength=-1.0",
            "--orbit-speed=0.15",
            "--gravity-well=1.0",
            "--black-holes=swift,very,really,basically"
        ])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
        output = ""
        for line in result.stdout.split('\n'):
            if "[DBG: Decoded" in line:
                output += line.split("'")[1].replace("\\n", "\n").replace("\\'", "'")
        return output.strip()
    except:
        return "[TIMEOUT]"

if __name__ == "__main__":
    if not os.path.exists(BINARY):
        print("Build binary first!")
        sys.exit(1)

    print("=" * 70)
    print("Baseline (plain Llama-3) vs Golden (Niodoo physics)")
    print("=" * 70)
    
    with open("artifacts/baseline_vs_golden.md", "w") as f:
        f.write("# Baseline vs Golden comparison\n\n")
        
        for label, prompt in PROMPTS:
            print(f"\n{'='*70}")
            print(f"{label}")
            print(f"Prompt: {prompt[:60]}...")
            print(f"{'='*70}")
            
            f.write(f"\n## {label}\n")
            f.write(f"**Prompt:** {prompt}\n\n")
            
            # BASELINE
            print("\n📦 BASELINE (No Physics):")
            print("-" * 40)
            baseline = run_test(prompt, use_physics=False)
            print(baseline[:500] + "..." if len(baseline) > 500 else baseline)
            f.write("### Baseline (Vanilla Llama-3)\n```text\n")
            f.write(baseline)
            f.write("\n```\n\n")
            f.flush()
            
            # GOLDEN
            print("\n🌟 GOLDEN (Niodoo Physics):")
            print("-" * 40)
            golden = run_test(prompt, use_physics=True)
            print(golden[:500] + "..." if len(golden) > 500 else golden)
            f.write("### Golden (Niodoo Physics)\n```text\n")
            f.write(golden)
            f.write("\n```\n\n")
            f.flush()
    
    print("\n" + "=" * 70)
    print("DONE. Results saved to artifacts/baseline_vs_golden.md")
