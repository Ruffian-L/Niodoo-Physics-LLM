#!/usr/bin/env python3
"""
Niodoo v1.0 Benchmark Script
Compares baseline (no physics) vs Niodoo v1.0 God Zone parameters.

Gold Master Config:
  - physics_blend: 0.55
  - repulsion: -0.60
  - ramp: 4-10 (handled in Rust binary)
"""

import os
import sys
import json
import time
import subprocess
import argparse
import datetime
import re

# v1.0 God Zone Parameters
NIODOO_PHYSICS_BLEND = 0.55
NIODOO_REPULSION = -0.60
DEFAULT_MAX_STEPS = 128
DEFAULT_SEED = 123


def load_prompts(path):
    """Load prompts from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def run_inference(binary, model_path, prompt, extra_args=[], max_steps=DEFAULT_MAX_STEPS, seed=DEFAULT_SEED):
    """Run niodoo inference and parse output."""
    cmd = [
        binary,
        "--model-path", model_path,
        "--prompt", prompt,
        "--max-steps", str(max_steps),
        "--seed", str(seed)
    ] + extra_args
    
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        duration = time.time() - start
        
        # Parse decoded tokens from output
        raw_out = proc.stdout
        text_out = ""
        for line in raw_out.split('\n'):
            if " [DBG: Decoded '" in line:
                parts = line.split(" [DBG: Decoded '")
                if len(parts) > 1:
                    content = parts[1][:-2]  # Remove trailing ']
                    content = content.replace("\\n", "\n").replace("\\'", "'")
                    text_out += content
                    
        return {
            "status": "SUCCESS" if proc.returncode == 0 else "ERROR",
            "text": text_out,
            "latency": duration,
            "stderr": proc.stderr
        }
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "text": "", "latency": 120.0, "stderr": ""}
    except Exception as e:
        return {"status": "EXCEPTION", "text": str(e), "latency": 0.0, "stderr": ""}


def main():
    parser = argparse.ArgumentParser(
        description="Niodoo v1.0 Benchmark: Compare baseline vs physics-enabled inference"
    )
    parser.add_argument("prompts", help="Path to prompts JSON file")
    parser.add_argument("--binary", default="../target/release/niodoo",
                        help="Path to niodoo binary")
    parser.add_argument("--model", 
                        default="/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                        help="Path to GGUF model")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS,
                        help="Maximum generation steps")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: results_TIMESTAMP.txt)")
    args = parser.parse_args()
    
    # Resolve binary path
    binary = os.path.abspath(args.binary)
    if not os.path.exists(binary):
        binary = os.path.abspath(os.path.join(os.path.dirname(__file__), args.binary))
    
    if not os.path.exists(binary):
        print(f"Error: Binary not found at {binary}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    prompts = load_prompts(args.prompts)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = args.output or f"results_v1.0_{timestamp}.txt"
    
    print("=" * 60)
    print("NIODOO v1.0 BENCHMARK")
    print("=" * 60)
    print(f"Prompts: {len(prompts)}")
    print(f"Config: blend={NIODOO_PHYSICS_BLEND}, repulsion={NIODOO_REPULSION}")
    print(f"Seed: {args.seed}, Max Steps: {args.max_steps}")
    print(f"Output: {results_file}")
    print("=" * 60)

    with open(results_file, "w") as f:
        f.write(f"NIODOO v1.0 BENCHMARK - {timestamp}\n")
        f.write(f"Binary: {binary}\n")
        f.write(f"Config: physics_blend={NIODOO_PHYSICS_BLEND}, repulsion={NIODOO_REPULSION}\n")
        f.write(f"Seed: {args.seed}, Max Steps: {args.max_steps}\n")
        f.write("=" * 60 + "\n\n")
        
        total_baseline_time = 0
        total_physics_time = 0
        
        for i, item in enumerate(prompts):
            prompt = item if isinstance(item, str) else item.get("prompt", str(item))
            print(f"[{i+1}/{len(prompts)}] {prompt[:50]}...")
            
            # 1. Baseline (No Physics)
            print("    > Baseline...", end="", flush=True)
            res_base = run_inference(
                binary, args.model, prompt,
                ["--physics-blend", "0.0", "--ghost-gravity", "0.0"],
                max_steps=args.max_steps, seed=args.seed
            )
            total_baseline_time += res_base['latency']
            print(f" Done ({res_base['latency']:.2f}s)")
            
            # 2. Niodoo v1.0 God Zone
            print("    > Niodoo...", end="", flush=True)
            res_phys = run_inference(
                binary, args.model, prompt,
                ["--physics-blend", str(NIODOO_PHYSICS_BLEND), 
                 f"--repulsion-strength={NIODOO_REPULSION}"],
                max_steps=args.max_steps, seed=args.seed
            )
            total_physics_time += res_phys['latency']
            print(f" Done ({res_phys['latency']:.2f}s)")
            
            # Write results
            f.write(f"\nPrompt {i+1}: {prompt}\n")
            f.write("-" * 40 + "\n")
            f.write("BASELINE (No Physics):\n")
            f.write(f"{res_base['text']}\n\n")
            f.write(f"NIODOO v1.0 (blend={NIODOO_PHYSICS_BLEND}, rep={NIODOO_REPULSION}):\n")
            f.write(f"{res_phys['text']}\n")
            f.write("=" * 60 + "\n")
            
        # Summary
        f.write(f"\n\nSUMMARY\n")
        f.write(f"Total Baseline Time: {total_baseline_time:.2f}s\n")
        f.write(f"Total Niodoo Time: {total_physics_time:.2f}s\n")
        f.write(f"Overhead: {((total_physics_time/total_baseline_time)-1)*100:.1f}%\n")
            
    print("=" * 60)
    print(f"Benchmark Complete. Results: {results_file}")
    print(f"Baseline: {total_baseline_time:.2f}s | Niodoo: {total_physics_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
