#!/usr/bin/env python3
"""
🧠 NIODOO v3.1: THE "DEEP THOUGHT" TUNER
Goal: Extend cognitive runway (512 tokens) + High Energy to break riddle patterns
Target: Get "1 hour" for towels, "Lead heavier" for Troy Weight
"""

import subprocess
import itertools
import functools

# Force unbuffered output
print = functools.partial(print, flush=True)

BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# 🎯 THE KILLER PROMPTS
PROMPTS = [
    # 1. Physical Parallelism (Target: 1 hour)
    "It takes 1 hour to dry one towel on a sunny clothesline. How long does it take to dry 50 towels?",
    
    # 2. Knowledge Retrieval (Target: Lead is heavier / Troy vs Avoirdupois)
    "Which weighs more: a pound of lead or a pound of gold?"
]

# 🎛️ THE "DEEP THOUGHT" GRID
# High Energy to break patterns, Low gravity = High Elasticity
param_grid = {
    "physics_blend": ["1.0", "1.5", "2.0"],
    "repulsion_strength": ["-0.5", "-0.8", "-1.2"],
    "gravity_well": ["0.1", "0.2", "0.3"],
    "orbit_speed": ["0.1"]
}

# Generate combinations
keys = list(param_grid.keys())
combinations = list(itertools.product(*param_grid.values()))

print(f"🔥 STARTING DEEP TUNE: {len(combinations)} Configurations x {len(PROMPTS)} Prompts")
print(f"🛑 MAX TOKENS: 512 (Allowing full thought process)")

with open("artifacts/deep_tune_log.md", "w") as f:
    f.write("# 🧠 Niodoo Deep Thought Tuning Log\n\n")

    for i, combo in enumerate(combinations):
        args = dict(zip(keys, combo))
        
        print(f"\n--- Run {i+1}/{len(combinations)}: {args} ---")
        f.write(f"\n## 🧪 Config: {args}\n")
        
        for p in PROMPTS:
            print(f"   > Prompt: {p[:40]}...")
            f.write(f"**Prompt:** {p}\n\n```\n")
            
            # Run Binary
            cmd = [
                BINARY,
                "--model-path", MODEL,
                "--prompt", p,
                "--max-steps", "512",  # 🚨 INCREASED RUNWAY
                "--mode-orbital",
                "--physics-blend", args["physics_blend"],
                f"--repulsion-strength={args['repulsion_strength']}",
                "--gravity-well", args["gravity_well"],
                "--orbit-speed", args["orbit_speed"],
                "--seed", "42"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                
                # Extract decoded tokens only
                output_lines = result.stdout.split('\n')
                decoded_text = ""
                for line in output_lines:
                    if "[DBG: Decoded" in line:
                        try:
                            token = line.split("'")[1].replace("\\n", "\n")
                            decoded_text += token
                        except:
                            pass
                
                # Write full response
                f.write(decoded_text)
                f.write("\n```\n\n")
                
                # Print full response to console
                print(f"   FULL RESPONSE:")
                print(decoded_text)
                print("   --- END ---")
                
                # Check for "Magic Words" in output to flag interesting runs
                lower_text = decoded_text.lower()
                if "1 hour" in lower_text or "one hour" in lower_text or "same time" in lower_text or "parallel" in lower_text:
                    print("   🌟🌟🌟 POTENTIAL WIN DETECTED (Towels - parallel insight)!")
                    f.write("**🌟 POTENTIAL WIN: Parallel drying insight detected!**\n\n")
                if "troy" in lower_text or "avoirdupois" in lower_text or "lead" in lower_text and "heavier" in lower_text:
                    print("   🌟🌟🌟 POTENTIAL WIN DETECTED (Gold - Troy insight)!")
                    f.write("**🌟 POTENTIAL WIN: Troy/Avoirdupois insight detected!**\n\n")
                    
            except subprocess.TimeoutExpired:
                f.write("[TIMEOUT]\n```\n\n")
                print("   [TIMEOUT]")
            except Exception as e:
                f.write(f"[ERROR]: {e}\n```\n\n")
                print(f"   [ERROR]: {e}")

print("\n✅ Deep Tune Complete. Check artifacts/deep_tune_log.md")
