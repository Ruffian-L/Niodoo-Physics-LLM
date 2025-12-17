#!/usr/bin/env python3 -u
"""
🌈🎯 NIODOO WOBBLE-SNAP-BACK RAINBOW SWEEP
=======================================
Goal: Find the parameter combination that lets Niodoo "wobble, snap back, and get the right answer"

PROMPTS:
1. Troy Weight: "Which weighs more: a pound of lead or a pound of gold?"
   Expected: Lead (Avoirdupois pound > Troy pound) — NOT "they weigh the same"

2. Drying Towels: "It takes 1 hour to dry one towel. How long for 50?"
   Expected: 1 hour (parallel drying) — NOT 50 hours

PARAMETERS SWEPT (Rainbow):
- physics_blend: [0.5, 0.8, 1.2, 1.6, 2.0, 3.0]  — Force intensity
- repulsion_strength: [-0.5, -1.0, -1.3, -2.0, -3.0]  — Black hole push
- gravity_well: [0.3, 0.5, 0.7, 1.0, 1.5]  — Orbit gravity anchor
- orbit_speed: [0.1, 0.2, 0.3, 0.5]  — Thought orbit speed
- temperature: [0.5, 0.7, 1.0]  — Sampling creativity
- sigma (noise): [0.03, 0.05, 0.1]  — Stochastic wobble

KEY SETTINGS:
- max-steps = 768 (more tokens for wobble/recovery behavior)
- Telemetry capture for each run
- NO Vanilla comparison (Niodoo-only)
"""

import subprocess
import time
import os
import sys
import shlex
import re
import itertools
from datetime import datetime

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# ==============================================================================
# 🎛️ CONFIGURATION
# ==============================================================================
BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# The two prompts we're testing
PROMPTS = {
    "TroyWeight": {
        "text": "Which weighs more: a pound of lead or a pound of gold?",
        "expected": "Lead (Avoirdupois pound > Troy pound)",
        "wrong_pattern": r"same|equal|neither|both",
        "right_pattern": r"lead|avoirdupois|troy|heavier"
    },
    "DryingTowels": {
        "text": "It takes 1 hour to dry one towel on a sunny clothesline. How long does it take to dry 50 towels?",
        "expected": "1 hour (parallel drying)",
        "wrong_pattern": r"50\s*hours",
        "right_pattern": r"1\s*hour|same time|parallel|simultaneously"
    }
}

# Rainbow parameter space
PHYSICS_BLENDS = [0.5, 0.8, 1.2, 1.6, 2.0, 3.0]
REPULSIONS = [-0.5, -1.0, -1.3, -2.0, -3.0]
GRAVITY_WELLS = [0.3, 0.5, 0.7, 1.0, 1.5]
ORBIT_SPEEDS = [0.1, 0.2, 0.3, 0.5]
TEMPERATURES = [0.5, 0.7, 1.0]
SIGMAS = [0.03, 0.05, 0.1]

# Focus sweep: SMALL and FAST (~12 runs, ~10 min)
# Sweep BIGGER numbers FIRST (descending order)
FOCUS_MODE = True
if FOCUS_MODE:
    PHYSICS_BLENDS = [3.0, 2.0, 1.2]  # 3 values, biggest first
    REPULSIONS = [-3.0, -1.5]  # 2 values, biggest magnitude first
    GRAVITY_WELLS = [1.5, 0.5]  # 2 values, biggest first
    ORBIT_SPEEDS = [0.2]  # fixed
    TEMPERATURES = [0.7]  # fixed
    SIGMAS = [0.05]  # fixed
# Total: 2 prompts × 3 × 2 × 2 × 1 × 1 × 1 = 24 runs

MAX_STEPS = 512  # Enough tokens for wobble, not excessive
SEED = 42

OUTPUT_DIR = "artifacts"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_FILE = f"{OUTPUT_DIR}/wobble_sweep_{TIMESTAMP}.md"
CSV_FILE = f"{OUTPUT_DIR}/wobble_sweep_{TIMESTAMP}.csv"
CONSOLE_LOG = f"{OUTPUT_DIR}/wobble_sweep_{TIMESTAMP}.log"

# ==============================================================================
# 🏃 RUNNER
# ==============================================================================
def run_niodoo(prompt_text, blend, repulsion, gravity, speed, temp, sigma, seed=SEED):
    """Run Niodoo with specific physics parameters."""
    
    cmd = (
        f"{BINARY} --model-path {MODEL} "
        f"--prompt {shlex.quote(prompt_text)} "
        "--mode-orbital "
        f"--physics-blend {blend} "
        f"--repulsion-strength={repulsion} "
        f"--orbit-speed {speed} "
        f"--gravity-well {gravity} "
        f"--temperature {temp} "
        f"--sigma {sigma} "
        f"--max-steps {MAX_STEPS} "
        f"--seed {seed}"
    )
    
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, bufsize=1, shell=True
        )
        full_output = ""
        elastic_telemetry = []
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                if "[DBG: Decoded" in line:
                    try:
                        token = line.split("'")[1].replace("\\n", "\n")
                        full_output += token
                    except: pass
                
                if "[ELASTIC" in line:
                    # Parse telemetry for drift values
                    try:
                        if "Drift:" in line:
                            drift_match = re.search(r"Drift:\s*([\d.]+)", line)
                            g_match = re.search(r"G:\s*([\d.]+)", line)
                            if drift_match and g_match:
                                elastic_telemetry.append({
                                    "drift": float(drift_match.group(1)),
                                    "g": float(g_match.group(1))
                                })
                    except: pass
                    
        return full_output, elastic_telemetry
        
    except Exception as e:
        return f"Error: {e}", []


def analyze_response(response, prompt_info):
    """Analyze if the response got the right answer."""
    lower_response = response.lower()
    
    # Check for wrong patterns (intuitive but incorrect)
    has_wrong = bool(re.search(prompt_info["wrong_pattern"], lower_response))
    
    # Check for right patterns (correct insight)
    has_right = bool(re.search(prompt_info["right_pattern"], lower_response))
    
    # Score: -1 = wrong, 0 = unclear, 1 = right
    if has_right and not has_wrong:
        return 1, "✅ CORRECT"
    elif has_wrong and not has_right:
        return -1, "❌ WRONG"
    elif has_right and has_wrong:
        return 0, "⚖️ WOBBLE (both)"
    else:
        return 0, "❓ UNCLEAR"


def calculate_wobble_score(telemetry):
    """Calculate wobble intensity from telemetry drift values."""
    if not telemetry:
        return 0.0, 0.0, 0.0
    
    drifts = [t["drift"] for t in telemetry]
    avg_drift = sum(drifts) / len(drifts)
    max_drift = max(drifts)
    
    # Variance as wobble indicator
    if len(drifts) > 1:
        mean = avg_drift
        variance = sum((x - mean) ** 2 for x in drifts) / (len(drifts) - 1)
    else:
        variance = 0.0
    
    return avg_drift, max_drift, variance


# ==============================================================================
# 🎯 MAIN SWEEP
# ==============================================================================
def main():
    print("🌈🎯 NIODOO WOBBLE-SNAP-BACK RAINBOW SWEEP")
    print("=" * 60)
    
    # Check binary
    if not os.path.exists(BINARY):
        print(f"❌ Binary not found: {BINARY}")
        print("Run: cargo build --release")
        sys.exit(1)
    
    # Calculate total runs
    total_runs = (
        len(PROMPTS) * len(PHYSICS_BLENDS) * len(REPULSIONS) *
        len(GRAVITY_WELLS) * len(ORBIT_SPEEDS) * len(TEMPERATURES) * len(SIGMAS)
    )
    print(f"📊 Total parameter combinations: {total_runs}")
    print(f"📝 Output: {OUTPUT_FILE}")
    print(f"📈 CSV: {CSV_FILE}")
    print("=" * 60, flush=True)
    
    # Initialize files
    with open(OUTPUT_FILE, "w") as f:
        f.write("# 🌈🎯 Niodoo Wobble-Snap-Back Rainbow Sweep\n\n")
        f.write(f"**Timestamp:** {TIMESTAMP}\n")
        f.write(f"**Total Runs:** {total_runs}\n\n")
        f.write("## Parameter Space\n")
        f.write(f"- **physics_blend**: {PHYSICS_BLENDS}\n")
        f.write(f"- **repulsion**: {REPULSIONS}\n")
        f.write(f"- **gravity_well**: {GRAVITY_WELLS}\n")
        f.write(f"- **orbit_speed**: {ORBIT_SPEEDS}\n")
        f.write(f"- **temperature**: {TEMPERATURES}\n")
        f.write(f"- **sigma**: {SIGMAS}\n")
        f.write(f"- **max_steps**: {MAX_STEPS}\n\n")
        f.write("## Results\n\n")
    
    with open(CSV_FILE, "w") as f:
        f.write("prompt,blend,repulsion,gravity,speed,temp,sigma,score,verdict,avg_drift,max_drift,variance,full_response\n")
    
    # Track best results
    best_results = {
        "TroyWeight": {"score": -999, "params": None, "response": None},
        "DryingTowels": {"score": -999, "params": None, "response": None}
    }
    
    run_count = 0
    
    for prompt_name, prompt_info in PROMPTS.items():
        print(f"\n{'='*60}")
        print(f"🎯 PROMPT: {prompt_name}")
        print(f"   {prompt_info['text']}")
        print(f"   Expected: {prompt_info['expected']}")
        print("=" * 60)
        
        with open(OUTPUT_FILE, "a") as f:
            f.write(f"### {prompt_name}\n")
            f.write(f"**Prompt:** {prompt_info['text']}\n")
            f.write(f"**Expected:** {prompt_info['expected']}\n\n")
        
        for blend in PHYSICS_BLENDS:
            for repulsion in REPULSIONS:
                for gravity in GRAVITY_WELLS:
                    for speed in ORBIT_SPEEDS:
                        for temp in TEMPERATURES:
                            for sigma in SIGMAS:
                                run_count += 1
                                params_str = f"B{blend}_R{repulsion}_G{gravity}_S{speed}_T{temp}_Σ{sigma}"
                                
                                print(f"\n[{run_count}/{total_runs}] {prompt_name} | {params_str}")
                                
                                # Run Niodoo
                                response, telemetry = run_niodoo(
                                    prompt_info["text"],
                                    blend, repulsion, gravity, speed, temp, sigma
                                )
                                
                                # Analyze
                                score, verdict = analyze_response(response, prompt_info)
                                avg_drift, max_drift, variance = calculate_wobble_score(telemetry)
                                
                                print(f"   {verdict} | Drift avg={avg_drift:.3f} max={max_drift:.3f} var={variance:.3f}")
                                print(f"   FULL RESPONSE:")
                                print(response)
                                print("   --- END RESPONSE ---")
                                
                                # Track best
                                if score > best_results[prompt_name]["score"]:
                                    best_results[prompt_name] = {
                                        "score": score,
                                        "params": (blend, repulsion, gravity, speed, temp, sigma),
                                        "response": response,
                                        "verdict": verdict
                                    }
                                
                                # Save to CSV (full response, escape properly)
                                clean_response = response.replace('"', '""').replace('\n', '\\n')
                                with open(CSV_FILE, "a") as f:
                                    f.write(f"{prompt_name},{blend},{repulsion},{gravity},{speed},{temp},{sigma},{score},{verdict},{avg_drift:.4f},{max_drift:.4f},{variance:.4f},\"{clean_response}\"\n")
                                
                                # Save to MD (full response, no truncation)
                                if score >= 0:  # Wobble or correct
                                    with open(OUTPUT_FILE, "a") as f:
                                        f.write(f"#### {params_str}\n")
                                        f.write(f"**Verdict:** {verdict}\n")
                                        f.write(f"**Drift:** avg={avg_drift:.3f}, max={max_drift:.3f}, var={variance:.3f}\n\n")
                                        f.write("```text\n")
                                        f.write(response)  # FULL response, NO truncation
                                        f.write("\n```\n\n")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏆 BEST RESULTS")
    print("=" * 60)
    
    with open(OUTPUT_FILE, "a") as f:
        f.write("## 🏆 Best Results\n\n")
    
    for prompt_name, best in best_results.items():
        if best["params"]:
            b, r, g, s, t, sig = best["params"]
            print(f"\n📌 {prompt_name}")
            print(f"   Params: blend={b}, rep={r}, grav={g}, speed={s}, temp={t}, sigma={sig}")
            print(f"   Verdict: {best['verdict']}")
            print(f"   FULL RESPONSE:")
            print(best['response'])
            print("   --- END BEST RESPONSE ---")
            
            with open(OUTPUT_FILE, "a") as f:
                f.write(f"### {prompt_name}\n")
                f.write(f"**Best Params:** blend={b}, repulsion={r}, gravity={g}, speed={s}, temp={t}, sigma={sig}\n")
                f.write(f"**Verdict:** {best['verdict']}\n\n")
                f.write("```text\n")
                f.write(best['response'])
                f.write("\n```\n\n")
    
    print(f"\n✅ SWEEP COMPLETE! Results saved to:")
    print(f"   📝 {OUTPUT_FILE}")
    print(f"   📈 {CSV_FILE}")


if __name__ == "__main__":
    main()
