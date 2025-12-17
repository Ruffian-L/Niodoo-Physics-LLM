#!/usr/bin/env python3
"""
🏆 GOLDEN RATIO 50 - OPTIMIZED RUN
50 Prompts (5 per category) using optimized per-category physics settings.
Based on analysis of 2500+ runs.
"""

import subprocess
import time
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from killer_prompts import KILLER_PROMPTS, CATEGORIES

BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# OPTIMIZED CONFIGURATION (The "Golden Ratio")
# Repulsion -1.0 and Layers 18-31 were globally best.
GLOBAL_REPULSION = -1.0
GLOBAL_LAYERS_START = 18
GLOBAL_LAYERS_END = 31

# Per-Category Blend Optimization
CATEGORY_BLEND = {
    "math": 1.2,           # Logic needs stability (was 2.0)
    "multi_step": 1.5,     # Lowered from 2.5
    "chain_thought": 1.2,  # Lowered from 2.0
    "hallucination": 1.5,  # Lowered from 2.0
    "context": 1.2,        # Lowered from 1.5
    "instruction": 1.0,    # Precision required
    "ambiguity": 1.5,
    "spatial": 1.0,
    "semantic_trap": 1.0,
    "edge_case": 0.8
}

def get_category(idx):
    for cat, rng in CATEGORIES.items():
        if idx in rng: return cat
    return "unknown"

def run_test(prompt, blend, repulsion, layer_start, layer_end):
    cmd = [
        BINARY, "--model-path", MODEL, "--prompt", prompt,
        "--mode-orbital", f"--physics-blend={blend}", f"--repulsion-strength={repulsion}",
        f"--physics-start-layer={layer_start}", f"--physics-end-layer={layer_end}",
        "--max-steps", "512", "--seed", "42"
    ]
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        latency = time.time() - start
        
        text = ""
        debris = 0
        for line in result.stdout.split('\n'):
            if "[DBG: Decoded" in line:
                try: text += line.split("'")[1].replace("\\n", "\n")
                except: pass
            if "[DEBRIS]" in line: debris += 1
            
        return {"text": text, "debris": debris, "time": latency}
    except Exception as e:
        return {"text": f"[ERROR: {str(e)}]", "debris": 0, "time": 0}

def main():
    log_file = "artifacts/golden_ratio_50.md"
    os.makedirs("artifacts", exist_ok=True)
    
    print("🏆 GOLDEN RATIO 50 STARTED")
    print(f"Global Config: Rep={GLOBAL_REPULSION}, Layers={GLOBAL_LAYERS_START}-{GLOBAL_LAYERS_END}")
    
    with open(log_file, "w") as f:
        f.write("# 🏆 Golden Ratio 50 - Optimized Run\n\n")
        f.write(f"**Global Config:** Repulsion {GLOBAL_REPULSION}, Layers {GLOBAL_LAYERS_START}-{GLOBAL_LAYERS_END}\n\n")
        
        count = 0
        for cat, rng in CATEGORIES.items():
            blend = CATEGORY_BLEND.get(cat, 1.5)
            # Take first 5 prompts from each category
            prompts = [KILLER_PROMPTS[i] for i in list(rng)[:5]]
            
            f.write(f"## Category: {cat.upper()} (Blend {blend})\n\n")
            print(f"\n--- {cat.upper()} (Blend {blend}) ---")
            
            for p in prompts:
                count += 1
                r = run_test(p, blend, GLOBAL_REPULSION, GLOBAL_LAYERS_START, GLOBAL_LAYERS_END)
                
                print(f"[{count}/50] {p[:50]}...")
                
                f.write(f"### Q: {p}\n\n")
                f.write(f"**A:**\n```\n{r['text']}\n```\n")
                f.write(f"*Stats: {r['time']:.1f}s, Debris: {r['debris']}*\n\n")
                f.flush()

    print(f"\n🏁 DONE. Results: {log_file}")

if __name__ == "__main__":
    main()
