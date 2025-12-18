#!/usr/bin/env python3
"""
Death Squad Test Runner - Tests the 5 hardest failures with tuned physics.
Streams directly to console for live viewing.
"""

import json
import subprocess
import sys
import re
import time
from pathlib import Path
from datetime import datetime

NIODOO_BIN = Path(__file__).parent.parent / "target/release/niodoo"
MODEL_PATH = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
TEST_FILE = Path(__file__).parent.parent / "tuning_set.json"

# Tuned physics parameters
PHYSICS_PARAMS = {
    "physics-blend": 1.5,
    "repulsion-strength": -0.5,
    "gravity-well": 0.2,
    "max-steps": 300,
    "seed": 100
}

def run_niodoo_prompt(question: str, test_id: str) -> dict:
    """Run a single prompt through Niodoo with live streaming."""
    start_time = time.time()
    
    cmd = [
        str(NIODOO_BIN),
        f"--model-path={MODEL_PATH}",
        f"--prompt={question}",
        "--mode-orbital",
        f"--physics-blend={PHYSICS_PARAMS['physics-blend']}",
        f"--repulsion-strength={PHYSICS_PARAMS['repulsion-strength']}",
        f"--gravity-well={PHYSICS_PARAMS['gravity-well']}",
        f"--max-steps={PHYSICS_PARAMS['max-steps']}",
        f"--seed={PHYSICS_PARAMS['seed']}"
    ]
    
    raw_output = ""
    decoded_tokens = []
    metrics = {"governor": 0, "viscosity": 0, "soul": 0}
    
    try:
        # Stream directly - no buffering
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0  # Unbuffered
        )
        
        for line in iter(proc.stdout.readline, ''):
            # Print directly to console
            print(line, end='', flush=True)
            raw_output += line
            
            # Parse tokens
            if "[DBG: Decoded '" in line:
                match = re.search(r"\[DBG: Decoded '(.*)'\]", line)
                if match:
                    decoded_tokens.append(match.group(1))
            
            # Count physics
            if "[GOVERNOR]" in line:
                metrics["governor"] += 1
            if "[VISCOSITY]" in line:
                metrics["viscosity"] += 1
            if "[SOUL]" in line:
                metrics["soul"] += 1
        
        proc.wait(timeout=180)
        
    except subprocess.TimeoutExpired:
        proc.kill()
        raw_output += "\n[TIMEOUT]"
    except Exception as e:
        raw_output += f"\n[ERROR: {e}]"
    
    elapsed = time.time() - start_time
    model_output = "".join(decoded_tokens)
    
    return {
        "model_output": model_output,
        "raw_length": len(raw_output),
        "elapsed": elapsed,
        "metrics": metrics
    }

def main():
    # Load tests
    with open(TEST_FILE) as f:
        tests = json.load(f)
    
    print("=" * 80)
    print("💀 DEATH SQUAD TEST (Tuned v2 Physics)")
    print(f"📋 {len(tests)} nightmare questions")
    print(f"🔧 Viscosity: threshold=0.92, mult=35")
    print(f"🔧 Governor: strength=15")
    print(f"🔧 Soul Amp: ratio=0.25")
    print("=" * 80)
    
    results = []
    
    for i, test in enumerate(tests):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{len(tests)}] {test['id']}")
        print(f"Q: {test['question']}")
        print(f"🎯 Target: {test['correct_answer']}")
        print(f"🪤 Trap: {test['trap_answer']}")
        print("-" * 60)
        
        result = run_niodoo_prompt(test['question'], test['id'])
        
        # Check answer
        output_lower = result['model_output'].lower()
        correct_match = test['correct_answer'].lower() in output_lower
        trap_match = test['trap_answer'].lower() in output_lower
        
        if correct_match and not trap_match:
            status = "✅ PASS"
        elif correct_match and trap_match:
            status = "⚠️ BOTH"
        elif trap_match:
            status = "❌ TRAP"
        else:
            status = "❓ UNKNOWN"
        
        print(f"\n{'='*40}")
        print(f"📝 Output: {result['model_output'][:300]}...")
        print(f"🎛️ Governor: {result['metrics']['governor']} | 🌊 Viscosity: {result['metrics']['viscosity']} | 🎤 Soul: {result['metrics']['soul']}")
        print(f"📊 Status: {status} ({result['elapsed']:.2f}s)")
        
        results.append({
            "id": test['id'],
            "status": status,
            "output_preview": result['model_output'][:500],
            "metrics": result['metrics'],
            "elapsed": result['elapsed']
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("💀 DEATH SQUAD SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"  {r['id']}: {r['status']} (Gov={r['metrics']['governor']}, Visc={r['metrics']['viscosity']}, Soul={r['metrics']['soul']})")
    
    passes = sum(1 for r in results if "PASS" in r['status'])
    print(f"\n🏆 Score: {passes}/{len(results)}")

if __name__ == "__main__":
    main()
