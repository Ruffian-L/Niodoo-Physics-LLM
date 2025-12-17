#!/usr/bin/env python3
"""
📊 NIODOO BASELINE BENCHMARK
Niodoo (Activation Steering) vs. Vanilla Top-P (Probabilistic Sampling)

This script generates the "Control Group" comparison data for the GitHub repo.
"""
import subprocess
import sys
import os

BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Test prompts covering different cognitive domains
PROMPTS = [
    ("🎨 CREATIVE", "Write a noir detective opening line set on a rainy Mars colony."),
    ("🦁 LOGIC", "If I put a coin in a cup and put the cup in the freezer, is the coin wet or dry? Explain step by step."),
    ("📝 PROSE", "Describe the feeling of watching a sunset from a spaceship window."),
]

def run_test(label, prompt, args):
    cmd = [
        BINARY, 
        "--model-path", MODEL, 
        "--prompt", prompt,
        "--seed", "42", 
        "--max-steps", "100"
    ] + args
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        output = ""
        for line in result.stdout.split('\n'):
            if "[DBG: Decoded" in line:
                output += line.split("'")[1].replace("\\n", "\n").replace("\\'", "'")
        return output.strip()
    except Exception as e:
        return f"[ERROR: {e}]"

def main():
    if not os.path.exists(BINARY):
        print("Build binary first: cargo build --release --bin niodoo")
        sys.exit(1)

    print("=" * 80)
    print("📊 NIODOO BASELINE BENCHMARK")
    print("Comparing Vanilla Llama-3.1 vs Niodoo Activation Steering")
    print("=" * 80)
    
    with open("artifacts/benchmark_results.md", "w") as f:
        f.write("# 📊 Niodoo Benchmark Results\n\n")
        f.write("## Vanilla Llama-3.1 vs Niodoo Activation Steering\n\n")
        f.write("**Model:** Meta-Llama-3.1-8B-Instruct (Q4_K_M)\n\n")
        f.write("**Niodoo Config:** Blend=1.0, Layers=16-31, Repulsion=-1.0\n\n")
        
        for label, prompt in PROMPTS:
            print(f"\n{'='*80}")
            print(f"{label}")
            print(f"Prompt: {prompt[:60]}...")
            print("="*80)
            
            f.write(f"\n## {label}\n")
            f.write(f"**Prompt:** {prompt}\n\n")
            
            # VANILLA (Physics off)
            print("\n📦 VANILLA (Top-P 0.9, Temp 0.7):")
            print("-" * 40)
            vanilla = run_test("Vanilla", prompt, [
                "--temperature", "0.7",
            ])
            print(vanilla)
            f.write("### 📦 Vanilla (No Physics)\n")
            f.write(f"```text\n{vanilla}\n```\n\n")
            f.flush()
            
            # NIODOO (Golden Config with Layer Banding)
            print("\n🌟 NIODOO (Golden Config + Layer Banding):")
            print("-" * 40)
            niodoo = run_test("Niodoo", prompt, [
                "--mode-orbital",
                "--physics-blend=1.0",
                "--repulsion-strength=-1.0",
                "--physics-start-layer=16",
                "--physics-end-layer=31",
            ])
            print(niodoo)
            f.write("### 🌟 Niodoo (Activation Steering)\n")
            f.write(f"```text\n{niodoo}\n```\n\n")
            f.flush()
    
    print("\n" + "=" * 80)
    print("DONE. Results saved to artifacts/benchmark_results.md")
    print("Copy the results to your README.md!")

if __name__ == "__main__":
    main()
