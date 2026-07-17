#!/usr/bin/env python3
"""
Hard-Prompt Stress Test (5000 runs)
5000 runs of hard prompts across all parameter combinations.
"""

import subprocess
import random
import time
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from failure_prompts import FAILURE_PROMPTS, CATEGORIES

BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

BLEND_RANGE = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
REPULSION_RANGE = [-0.5, -1.0, -1.5, -2.0, -2.5]
LAYER_START_RANGE = [10, 12, 14, 16, 18, 20]
LAYER_END_RANGE = [26, 28, 30, 31]

MAX_TOKENS = 80
TOTAL_RUNS = 5000

def run_test(prompt, blend, repulsion, layer_start, layer_end, seed):
    cmd = [
        BINARY, "--model-path", MODEL, "--prompt", prompt,
        "--mode-orbital", f"--physics-blend={blend}", f"--repulsion-strength={repulsion}",
        f"--physics-start-layer={layer_start}", f"--physics-end-layer={layer_end}",
        "--max-steps", str(MAX_TOKENS), "--seed", str(seed)
    ]
    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        latency = time.time() - start
        text = ""
        debris = 0
        for line in result.stdout.split('\n'):
            if "[DBG: Decoded" in line:
                try: text += line.split("'")[1].replace("\\n", "\n")
                except: pass
            if "[DEBRIS]" in line: debris += 1
        glitch = any(x in text for x in ['://', '.Forms', '#', 'assistant', 'Angeles', '�'])
        return {"ok": True, "text": text[:300], "debris": debris, "glitch": glitch, "words": len(text.split()), "time": latency}
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": "", "debris": 0, "glitch": True, "words": 0, "time": 90}
    except Exception as e:
        return {"ok": False, "text": str(e), "debris": 0, "glitch": True, "words": 0, "time": 0}

def get_category(idx):
    for cat, rng in CATEGORIES.items():
        if idx in rng: return cat
    return "unknown"

def main():
    log = "archive/experiment-logs/hard_prompts_5000_log.md"
    os.makedirs("artifacts", exist_ok=True)
    print("Hard-Prompt Stress Test (5000 runs) starting")
    
    with open(log, "w") as f:
        f.write(f"# Hard-Prompt Stress Test (5000 runs)\nStarted: {datetime.now()}\n\n")
        f.write("| # | Cat | Blend | Rep | Layers | Status | Debris | Words | Time |\n")
        f.write("|---|-----|-------|-----|--------|--------|--------|-------|------|\n")
        
        stats = {"ok": 0, "glitch": 0, "timeout": 0, "debris": 0}
        cat_stats = {c: {"ok": 0, "fail": 0} for c in CATEGORIES}
        
        for run in range(1, TOTAL_RUNS + 1):
            idx = random.randint(0, len(FAILURE_PROMPTS)-1)
            prompt = FAILURE_PROMPTS[idx]
            cat = get_category(idx)
            blend = random.choice(BLEND_RANGE)
            rep = random.choice(REPULSION_RANGE)
            ls = random.choice(LAYER_START_RANGE)
            le = random.choice(LAYER_END_RANGE)
            
            r = run_test(prompt, blend, rep, ls, le, random.randint(1, 9999))
            
            if r["ok"] and not r["glitch"]:
                stats["ok"] += 1
                cat_stats[cat]["ok"] += 1
            else:
                stats["glitch"] += 1
                cat_stats[cat]["fail"] += 1
            if not r["ok"]: stats["timeout"] += 1
            stats["debris"] += r["debris"]
            
            icon = "✅" if r["ok"] and not r["glitch"] else "⚠️"
            f.write(f"| {run} | {cat[:4]} | {blend} | {rep} | {ls}-{le} | {icon} | {r['debris']} | {r['words']} | {r['time']:.1f}s |\n")
            f.flush()
            
            print(f"[{run}/{TOTAL_RUNS}] {cat[:6]:6} B={blend} | {icon} D={r['debris']:2} | {prompt[:50]}...")
            
            if run % 100 == 0:
                rate = stats["ok"] / run * 100
                print(f"\n📊 CHECKPOINT {run}: Success={rate:.1f}% Glitch={stats['glitch']} Debris={stats['debris']}")
                f.write(f"\n**CP {run}:** OK={stats['ok']} Fail={stats['glitch']} Rate={rate:.1f}%\n\n")
        
        f.write(f"\n## FINAL: OK={stats['ok']} Fail={stats['glitch']} Rate={stats['ok']/TOTAL_RUNS*100:.1f}%\n")
        for c, s in cat_stats.items():
            f.write(f"- {c}: {s['ok']}/{s['ok']+s['fail']} ({s['ok']/(s['ok']+s['fail']+0.001)*100:.0f}%)\n")
    
    print(f"\n🏁 DONE. Results: {log}")

if __name__ == "__main__":
    main()
