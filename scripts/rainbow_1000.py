#!/usr/bin/env python3
"""
🌈 RAINBOW MICRO TUNE - 1000 PROMPT RUNNER
Continuously runs prompts across blend settings to find micro-optimal parameters.
Outputs to console and logs results for analysis.

Usage:
    python3 scripts/rainbow_1000.py &
    tail -f artifacts/rainbow_1000_log.md
"""

import subprocess
import random
import time
import json
import os
import sys
from datetime import datetime

# Add benchmarks to path for prompt bank
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from prompt_bank_200 import PROMPTS

BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# Rainbow parameter ranges for micro-tuning
BLEND_RANGE = [0.8, 1.0, 1.2, 1.5, 2.0]
REPULSION_RANGE = [-0.5, -1.0, -1.5, -2.0]
LAYER_START_RANGE = [12, 14, 16, 18, 20]
LAYER_END_RANGE = [28, 30, 31]

MAX_TOKENS = 60
TOTAL_RUNS = 1000

def run_test(prompt, blend, repulsion, layer_start, layer_end, seed):
    cmd = [
        BINARY,
        "--model-path", MODEL,
        "--prompt", prompt,
        "--mode-orbital",
        f"--physics-blend={blend}",
        f"--repulsion-strength={repulsion}",
        f"--physics-start-layer={layer_start}",
        f"--physics-end-layer={layer_end}",
        "--max-steps", str(MAX_TOKENS),
        "--seed", str(seed)
    ]
    
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        latency = time.time() - start
        
        # Parse output
        text = ""
        debris_count = 0
        for line in result.stdout.split('\n'):
            if "[DBG: Decoded" in line:
                try:
                    text += line.split("'")[1].replace("\\n", "\n")
                except:
                    pass
            if "[DEBRIS]" in line:
                debris_count += 1
        
        # Quality metrics
        has_glitch = any(x in text for x in ['://', '.Forms', '#', 'assistant', 'Angeles'])
        word_count = len(text.split())
        
        return {
            "status": "OK",
            "text": text[:200],
            "debris": debris_count,
            "glitch": has_glitch,
            "words": word_count,
            "latency": latency
        }
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "text": "", "debris": 0, "glitch": True, "words": 0, "latency": 60}
    except Exception as e:
        return {"status": "ERROR", "text": str(e), "debris": 0, "glitch": True, "words": 0, "latency": 0}

def main():
    if not os.path.exists(BINARY):
        print("Build binary first: cargo build --release --bin niodoo")
        sys.exit(1)
    
    log_path = "artifacts/rainbow_1000_log.md"
    os.makedirs("artifacts", exist_ok=True)
    
    print("=" * 80)
    print("🌈 RAINBOW MICRO TUNE - 1000 PROMPT RUNNER")
    print(f"Logging to: {log_path}")
    print("=" * 80)
    
    with open(log_path, "w") as f:
        f.write("# 🌈 Rainbow 1000 Micro Tune Log\n\n")
        f.write(f"Started: {datetime.now().isoformat()}\n\n")
        f.write("| Run | Blend | Rep | Layers | Prompt | Status | Debris | Glitch | Words | Time |\n")
        f.write("|-----|-------|-----|--------|--------|--------|--------|--------|-------|------|\n")
        f.flush()
        
        stats = {"ok": 0, "glitch": 0, "timeout": 0, "total_debris": 0}
        
        for run in range(1, TOTAL_RUNS + 1):
            # Random parameters
            blend = random.choice(BLEND_RANGE)
            repulsion = random.choice(REPULSION_RANGE)
            layer_start = random.choice(LAYER_START_RANGE)
            layer_end = random.choice(LAYER_END_RANGE)
            prompt = random.choice(PROMPTS)
            seed = random.randint(1, 10000)
            
            # Run test
            result = run_test(prompt, blend, repulsion, layer_start, layer_end, seed)
            
            # Update stats
            if result["status"] == "OK":
                stats["ok"] += 1
                if result["glitch"]:
                    stats["glitch"] += 1
            else:
                stats["timeout"] += 1
            stats["total_debris"] += result["debris"]
            
            # Log
            status_icon = "✅" if result["status"] == "OK" and not result["glitch"] else "⚠️" if result["glitch"] else "❌"
            prompt_short = prompt[:30].replace("|", "").replace("\n", " ")
            layers_str = f"{layer_start}-{layer_end}"
            
            log_line = f"| {run} | {blend} | {repulsion} | {layers_str} | {prompt_short}... | {status_icon} | {result['debris']} | {result['glitch']} | {result['words']} | {result['latency']:.1f}s |\n"
            f.write(log_line)
            f.flush()
            
            # Console output
            print(f"[{run}/{TOTAL_RUNS}] B={blend} R={repulsion} L={layers_str} | {status_icon} D={result['debris']} G={result['glitch']} | {prompt_short[:40]}...")
            
            # Periodic summary
            if run % 50 == 0:
                success_rate = (stats["ok"] - stats["glitch"]) / run * 100
                print(f"\n📊 CHECKPOINT @ {run}: Success={success_rate:.1f}% Glitches={stats['glitch']} Debris={stats['total_debris']}\n")
                f.write(f"\n**Checkpoint {run}:** Success={success_rate:.1f}%, Glitches={stats['glitch']}, Debris={stats['total_debris']}\n\n")
                f.flush()
        
        # Final summary
        f.write(f"\n## Final Stats\n")
        f.write(f"- Total Runs: {TOTAL_RUNS}\n")
        f.write(f"- Successful: {stats['ok']}\n")
        f.write(f"- Glitches: {stats['glitch']}\n")
        f.write(f"- Timeouts: {stats['timeout']}\n")
        f.write(f"- Total Debris Blocked: {stats['total_debris']}\n")
        f.write(f"- Success Rate: {(stats['ok'] - stats['glitch']) / TOTAL_RUNS * 100:.1f}%\n")
    
    print("\n" + "=" * 80)
    print("🏁 RAINBOW 1000 COMPLETE")
    print(f"Results: {log_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
