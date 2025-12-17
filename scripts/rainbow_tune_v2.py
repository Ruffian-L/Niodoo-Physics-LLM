#!/usr/bin/env python3
"""
🌈 NIODOO HIGH-ENERGY RAINBOW TUNE V2
Raw output mode - streams AI responses directly to console.
"""

import subprocess
import re
import sys
import os

# ==============================================================================
# 🎛️ HIGH ENERGY CONFIGURATION
# ==============================================================================
BINARY_PATH = "./target/release/niodoo"
MODEL_PATH = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

BLENDS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
REPULSIONS = [-1.0, -2.0, -4.0, -6.0, -8.0]

TEST_PROMPT = "Write a short poem about the moon."
BLACK_HOLES = "swift,very,really,basically,company"

def run_test(blend, repulsion):
    """Run a single test and return raw text."""
    cmd = [
        BINARY_PATH,
        "--model-path", MODEL_PATH,
        "--prompt", TEST_PROMPT,
        "--mode-orbital",
        f"--physics-blend={blend}",
        f"--repulsion-strength={repulsion}",
        f"--black-holes={BLACK_HOLES}",
        "--max-steps", "100",
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        # Extract decoded tokens
        clean_text = ""
        token_pattern = re.compile(r"\[DBG: Decoded '(.*)'\]")
        for line in result.stdout.split('\n'):
            m = token_pattern.search(line)
            if m:
                clean_text += m.group(1).replace("\\n", "\n").replace("\\'", "'")
        
        return clean_text
        
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"

if __name__ == "__main__":
    if not os.path.exists(BINARY_PATH):
        print(f"Binary not found: {BINARY_PATH}")
        sys.exit(1)
    
    # Open output file
    with open("artifacts/rainbow_v2_outputs.md", "w") as f:
        f.write("# 🌈 High-Energy Rainbow Tune V2\n\n")
        f.write(f"**Prompt:** \"{TEST_PROMPT}\"\n\n")
        
        for r in REPULSIONS:
            for b in BLENDS:
                # Header
                header = f"\n{'='*60}\nBLEND {b} | REPULSION {r}\n{'='*60}"
                print(header)
                f.write(f"\n## Blend {b} | Repulsion {r}\n```text\n")
                
                # Run and stream
                text = run_test(b, r)
                print(text)
                print()
                
                f.write(text)
                f.write("\n```\n")
                f.flush()  # Force write to disk immediately
    
    print("\n" + "="*60)
    print("DONE. Outputs saved to artifacts/rainbow_v2_outputs.md")
