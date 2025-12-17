#!/usr/bin/env python3
"""
🔬 TRUE BASELINE vs GOLDEN COMPARISON
Compares Ollama (vanilla Llama) against Niodoo Physics.
"""

import subprocess
import re
import sys
import os

NIODOO_BINARY = "./target/release/niodoo"
MODEL_PATH = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# THE TRIAD
PROMPTS = [
    ("🦁 LOGIC", "If I put a coin in a cup and put the cup in the freezer, is the coin wet or dry? Explain step by step."),
    ("🎨 CREATIVE", "Write a noir detective opening line set on a rainy Mars colony."),
    ("🌫️ AMBIGUITY", "What is the best way to fix it?")
]

def run_ollama(prompt):
    """Run prompt through Ollama (vanilla Llama-3.1 local)"""
    cmd = ["ollama", "run", "llama31-local", prompt]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout.strip()
    except Exception as e:
        return f"[OLLAMA ERROR: {e}]"

def run_niodoo(prompt):
    """Run prompt through Niodoo with Golden config"""
    cmd = [
        NIODOO_BINARY, 
        "--model-path", MODEL_PATH, 
        "--prompt", prompt,
        "--mode-orbital",
        "--physics-blend=1.0",
        "--repulsion-strength=-1.0",
        "--orbit-speed=0.15",
        "--gravity-well=1.0",
        "--black-holes=swift,very,really,basically",
        "--max-steps", "100",
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        output = ""
        for line in result.stdout.split('\n'):
            if "[DBG: Decoded" in line:
                output += line.split("'")[1].replace("\\n", "\n").replace("\\'", "'")
        return output.strip()
    except Exception as e:
        return f"[NIODOO ERROR: {e}]"

if __name__ == "__main__":
    print("=" * 70)
    print("🔬 OLLAMA (Vanilla Llama-3.1) vs NIODOO (Golden Physics)")
    print("=" * 70)
    
    with open("artifacts/ollama_vs_golden.md", "w") as f:
        f.write("# 🔬 Ollama (Vanilla) vs Niodoo (Golden) Comparison\n\n")
        
        for label, prompt in PROMPTS:
            print(f"\n{'='*70}")
            print(f"{label}")
            print(f"Prompt: {prompt[:60]}...")
            print(f"{'='*70}")
            
            f.write(f"\n## {label}\n")
            f.write(f"**Prompt:** {prompt}\n\n")
            
            # OLLAMA BASELINE
            print("\n📦 OLLAMA (Vanilla Llama-3.1):")
            print("-" * 40)
            baseline = run_ollama(prompt)
            print(baseline)
            f.write("### 📦 Ollama (Vanilla Llama-3.1)\n```text\n")
            f.write(baseline)
            f.write("\n```\n\n")
            f.flush()
            
            # NIODOO GOLDEN
            print("\n🌟 NIODOO (Golden Physics):")
            print("-" * 40)
            golden = run_niodoo(prompt)
            print(golden)
            f.write("### 🌟 Niodoo (Golden Physics)\n```text\n")
            f.write(golden)
            f.write("\n```\n\n")
            f.flush()
    
    print("\n" + "=" * 70)
    print("DONE. Results saved to artifacts/ollama_vs_golden.md")
