#!/usr/bin/env python3
"""
Niodoo v2.0 Benchmark Script
Shows LIVE AI responses comparing Baseline vs Niodoo Activation Steering.
"""

import os
import sys
import json
import time
import subprocess
import argparse
import datetime

# v2.0 Validated Configuration
NIODOO_PHYSICS_BLEND = 1.0
NIODOO_REPULSION = -1.0
NIODOO_START_LAYER = 16
NIODOO_END_LAYER = 31
DEFAULT_MAX_STEPS = 128
DEFAULT_SEED = 123


def load_prompts(path):
    with open(path, 'r') as f:
        return json.load(f)


def run_inference(binary, model_path, prompt, extra_args=[], max_steps=DEFAULT_MAX_STEPS, seed=DEFAULT_SEED):
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
        
        # Parse decoded tokens - format is: [DBG: Decoded 'token']
        text_out = ""
        for line in proc.stderr.split('\n') + proc.stdout.split('\n'):
            if "[DBG: Decoded '" in line:
                try:
                    start_idx = line.index("[DBG: Decoded '") + len("[DBG: Decoded '")
                    end_idx = line.rindex("']")
                    if end_idx > start_idx:
                        content = line[start_idx:end_idx]
                        content = content.replace("\\n", "\n").replace("\\'", "'")
                        text_out += content
                except ValueError:
                    pass
                    
        return {"status": "OK", "text": text_out, "latency": duration}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "text": "", "latency": 120.0}
    except Exception as e:
        return {"status": "ERROR", "text": str(e), "latency": 0.0}


def main():
    parser = argparse.ArgumentParser(description="Niodoo v2.0 Benchmark - LIVE OUTPUT")
    parser.add_argument("prompts", help="Path to prompts JSON")
    parser.add_argument("--binary", default="../target/release/niodoo")
    parser.add_argument("--model", default="/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    
    binary = os.path.abspath(args.binary)
    if not os.path.exists(binary):
        binary = os.path.abspath(os.path.join(os.path.dirname(__file__), args.binary))
    
    if not os.path.exists(binary):
        print(f"ERROR: Binary not found at {binary}")
        sys.exit(1)

    prompts = load_prompts(args.prompts)
    
    print("=" * 70)
    print("NIODOO v2.0 BENCHMARK - Baseline vs Activation Steering")
    print(f"Config: blend={NIODOO_PHYSICS_BLEND}, repulsion={NIODOO_REPULSION}, layers={NIODOO_START_LAYER}-{NIODOO_END_LAYER}")
    print("=" * 70)
    
    results = []
    
    for i, item in enumerate(prompts):
        prompt = item if isinstance(item, str) else item.get("prompt", str(item))
        
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(prompts)}] {prompt}")
        print("=" * 70)
        
        # BASELINE (No Physics)
        print("\n--- BASELINE (No Physics) ---")
        res_base = run_inference(
            binary, args.model, prompt,
            [],  # No extra args = vanilla mode
            max_steps=args.max_steps, seed=args.seed
        )
        print(res_base['text'] if res_base['text'] else "[No output captured]")
        
        # NIODOO v2.0
        print(f"\n--- NIODOO v2.0 (blend={NIODOO_PHYSICS_BLEND}, layers={NIODOO_START_LAYER}-{NIODOO_END_LAYER}) ---")
        res_phys = run_inference(
            binary, args.model, prompt,
            [
                "--mode-orbital",
                f"--physics-blend={NIODOO_PHYSICS_BLEND}",
                f"--repulsion-strength={NIODOO_REPULSION}",
                f"--physics-start-layer={NIODOO_START_LAYER}",
                f"--physics-end-layer={NIODOO_END_LAYER}",
            ],
            max_steps=args.max_steps, seed=args.seed
        )
        print(res_phys['text'] if res_phys['text'] else "[No output captured]")
        
        results.append({
            "prompt": prompt,
            "baseline": res_base['text'],
            "niodoo": res_phys['text']
        })
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            f.write(f"NIODOO v2.0 BENCHMARK\n")
            f.write(f"Config: blend={NIODOO_PHYSICS_BLEND}, repulsion={NIODOO_REPULSION}, layers={NIODOO_START_LAYER}-{NIODOO_END_LAYER}\n")
            f.write("=" * 70 + "\n\n")
            for r in results:
                f.write(f"PROMPT: {r['prompt']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"BASELINE:\n{r['baseline']}\n\n")
                f.write(f"NIODOO:\n{r['niodoo']}\n")
                f.write("=" * 70 + "\n\n")
        print(f"\nResults saved to: {args.output}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

