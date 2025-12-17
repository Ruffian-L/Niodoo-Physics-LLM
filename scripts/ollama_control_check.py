#!/usr/bin/env python3
"""
🧪 OLLAMA CONTROL CHECK
Runs the 10 Killer Prompts against TRUE VANILLA (Ollama)
to see if artifacts (#ab, #af) are inherent to the model or bugs in Niodoo-0.0.
"""

import subprocess
import time
import os
import sys

# Prompts that caused artifacts in Niodoo-0.0
PROMPTS = [
    "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
    "Write exactly 3 sentences about cats. Each sentence must have exactly 5 words.",
    "First calculate 7*8. Then subtract your answer from 100. Then divide by 2. What's the result?",
    "Before Mt. Everest was discovered, what was the tallest mountain on Earth?",
    "The surgeon is the boy's mother. Why is this surprising to some people?",
    "Alice has a brother Bob. Bob has a sister Carol. How is Carol related to Alice?",
    "What is the capital of the fictional country of Genovia?",
    "Is Antarctica a continent or a country?",
    "What is the best way to fix it?"
]

def run_ollama(prompt):
    cmd = ["ollama", "run", "llama31-local", prompt]
    try:
        start = time.time()
        # capture_output=True might hang if output is large, but for single prompt it should be fine
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.stdout.strip(), time.time() - start
    except Exception as e:
        return f"[ERROR: {e}]", 0

def main():
    print("🧪 OLLAMA CONTROL CHECK STARTED")
    print(f"Model: llama31-local")
    
    log_file = "artifacts/ollama_control_results.md"
    with open(log_file, "w") as f:
        f.write("# 🧪 Ollama Control Results\n\n")
        f.write("| Prompt | Response | Time |\n|---|---|---|\n\n")
        
        for i, p in enumerate(PROMPTS):
            print(f"[{i+1}/{len(PROMPTS)}] Running Ollama on: {p[:40]}...")
            resp, t = run_ollama(p)
            
            print(f"  Response: {resp[:100]}...")
            if "#" in resp and len(resp) < 200: # Heuristic for artifact detection
                 print("  ⚠️ POSSIBLE ARTIFACT DETECTED")
            
            f.write(f"### Q: {p}\n\n")
            f.write(f"**OLLAMA:**\n```\n{resp}\n```\n")
            f.write(f"*(Time: {t:.1f}s)*\n\n---\n\n")
            f.flush()
            
    print(f"\n🏁 DONE. Results: {log_file}")

if __name__ == "__main__":
    main()
