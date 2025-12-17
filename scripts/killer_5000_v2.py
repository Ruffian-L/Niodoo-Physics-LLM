#!/usr/bin/env python3
"""
💀🌈 KILLER RAINBOW 5000 V2 - WITH FULL AI RESPONSES
Same test but saves COMPLETE AI output for review.
"""

import subprocess
import random
import time
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'benchmarks'))
from killer_prompts import KILLER_PROMPTS, CATEGORIES

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
        return {"ok": True, "text": text, "debris": debris, "glitch": glitch, "words": len(text.split()), "time": latency}
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": "[TIMEOUT]", "debris": 0, "glitch": True, "words": 0, "time": 90}
    except Exception as e:
        return {"ok": False, "text": f"[ERROR: {e}]", "debris": 0, "glitch": True, "words": 0, "time": 0}

def get_category(idx):
    for cat, rng in CATEGORIES.items():
        if idx in rng: return cat
    return "unknown"

def main():
    log = "artifacts/killer_5000_FULL.md"
    os.makedirs("artifacts", exist_ok=True)
    print("💀🌈 KILLER RAINBOW 5000 V2 - FULL RESPONSES")
    
    with open(log, "w") as f:
        f.write(f"# 💀 Killer Rainbow 5000 - FULL AI RESPONSES\n")
        f.write(f"Started: {datetime.now()}\n\n")
        
        stats = {"ok": 0, "glitch": 0, "timeout": 0, "debris": 0}
        cat_stats = {c: {"ok": 0, "fail": 0} for c in CATEGORIES}
        
        for run in range(1, TOTAL_RUNS + 1):
            idx = random.randint(0, len(KILLER_PROMPTS)-1)
            prompt = KILLER_PROMPTS[idx]
            cat = get_category(idx)
            blend = random.choice(BLEND_RANGE)
            rep = random.choice(REPULSION_RANGE)
            ls = random.choice(LAYER_START_RANGE)
            le = random.choice(LAYER_END_RANGE)
            seed = random.randint(1, 9999)
            
            r = run_test(prompt, blend, rep, ls, le, seed)
            
            if r["ok"] and not r["glitch"]:
                stats["ok"] += 1
                cat_stats[cat]["ok"] += 1
                icon = "✅"
            else:
                stats["glitch"] += 1
                cat_stats[cat]["fail"] += 1
                icon = "⚠️"
            if not r["ok"]: stats["timeout"] += 1
            stats["debris"] += r["debris"]
            
            # WRITE FULL RESPONSE TO LOG
            f.write(f"\n---\n\n")
            f.write(f"## Run {run}/5000 {icon}\n\n")
            f.write(f"**Category:** {cat} | **Blend:** {blend} | **Repulsion:** {rep} | **Layers:** {ls}-{le}\n\n")
            f.write(f"**Prompt:** {prompt}\n\n")
            f.write(f"**AI Response:**\n```\n{r['text']}\n```\n\n")
            f.write(f"**Stats:** Debris={r['debris']} | Words={r['words']} | Time={r['time']:.1f}s | Glitch={r['glitch']}\n")
            f.flush()
            
            # Console output
            print(f"[{run}/{TOTAL_RUNS}] {cat[:6]:6} B={blend} | {icon} D={r['debris']:2}")
            print(f"  Q: {prompt}...")
            print(f"  A: {r['text']}...")
            print()
            
            if run % 100 == 0:
                rate = stats["ok"] / run * 100
                print(f"\n📊 CP {run}: OK={stats['ok']} ({rate:.1f}%) Debris={stats['debris']}\n")
                f.write(f"\n### Checkpoint {run}\nSuccess Rate: {rate:.1f}%\n\n")
        
        f.write(f"\n---\n\n## FINAL RESULTS\n")
        f.write(f"- **Success:** {stats['ok']}/{TOTAL_RUNS} ({stats['ok']/TOTAL_RUNS*100:.1f}%)\n")
        f.write(f"- **Glitches:** {stats['glitch']}\n")
        f.write(f"- **Debris Blocked:** {stats['debris']}\n\n")
        f.write(f"### Per-Category Breakdown\n")
        for c, s in cat_stats.items():
            total = s['ok'] + s['fail']
            rate = s['ok'] / (total + 0.001) * 100
            f.write(f"- **{c}:** {s['ok']}/{total} ({rate:.0f}%)\n")
    
    print(f"\n🏁 DONE. Full results: {log}")

if __name__ == "__main__":
    main()
