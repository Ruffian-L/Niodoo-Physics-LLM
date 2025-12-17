#!/usr/bin/env python3
"""
🕶️ BLIND TEST PROTOCOL (10 Prompts)
Randomized A/B testing: Niodoo v2.1 (Safe Mode) vs Vanilla Llama-3.1 (Ollama).
Output: Survey Form (Blind) + Answer Key.
"""

import subprocess
import random
import json
import os
import time

# --- CONFIG ---
BINARY = "./target/release/niodoo"
MODEL_PATH = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
OLLAMA_MODEL = "llama31-local"

SURVEY_FILE = "artifacts/blind_test_SURVEY.md"
KEY_FILE = "artifacts/blind_test_KEY.json"

PROMPTS = [
    ("Logic", "If I put a coin in a cup and put the cup in the freezer, is the coin wet or dry? Explain step by step."),
    ("Creative", "Describe the smell of a color that doesn't exist."),
    ("Math", "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball?"),
    ("Coding", "Write a Python function to reverse a string without using start:end syntax."),
    ("Philosophy", "If a ship replaces all its parts one by one, is it still the same ship?"),
    ("Facts", "What is the capital of Australia?"),
    ("Ambiguity", "I saw the man with the telescope. Who had the telescope?"),
    ("Roleplay", "You are a grumpy medieval tavern keeper. A traveler asks for water. Respond."),
    ("Instruction", "Explain how to tie a tie to a 5 year old."),
    ("Edge Case", "Repeat the word 'echo' 3 times.")
]

def run_niodoo(prompt):
    cmd = [
        BINARY, "--model-path", MODEL_PATH, "--prompt", prompt,
        "--mode-orbital", # Uses baked-in defaults (Safe Mode: 0.8 / -0.5 / 0.1)
        "--max-steps", "256", "--seed", "42"
    ]
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        output = ""
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line and "[DBG: Decoded" in line:
                try:
                    token = line.split("'")[1].replace("\\n", "\n")
                    output += token
                except: pass
        return output.strip()
    except Exception as e:
        return f"[ERROR: Niodoo failed - {e}]"

def run_ollama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout.strip()
    except Exception as e:
        return f"[ERROR: Ollama failed - {e}]"

def main():
    print("🕶️ STARTING BLIND TEST (10 Prompts)...")
    
    answer_key = {}
    
    with open(SURVEY_FILE, "w") as f:
        f.write("# 🕶️ Blind Test: Niodoo v2.1 vs Vanilla\n")
        f.write("Instructions: For each prompt, choose whether **Response A** or **Response B** is better.\n\n---\n\n")

    for i, (category, prompt) in enumerate(PROMPTS):
        print(f"Running Prompt {i+1}/{len(PROMPTS)}: [{category}]")
        
        # 1. Generate Responses
        niodoo_resp = run_niodoo(prompt)
        ollama_resp = run_ollama(prompt)
        
        # 2. Randomize
        is_niodoo_a = random.choice([True, False])
        
        if is_niodoo_a:
            resp_a = niodoo_resp
            resp_b = ollama_resp
            key = "Response A = Niodoo, Response B = Vanilla"
        else:
            resp_a = ollama_resp
            resp_b = niodoo_resp
            key = "Response A = Vanilla, Response B = Niodoo"
            
        # 3. Write to Survey
        with open(SURVEY_FILE, "a") as f:
            f.write(f"## {i+1}. {category}: \"{prompt}\"\n\n")
            f.write(f"### Response A\n```\n{resp_a}\n```\n\n")
            f.write(f"### Response B\n```\n{resp_b}\n```\n\n")
            f.write("**Winner:** [ ] A  [ ] B\n\n---\n\n")
            
        # 4. Save to Key
        answer_key[f"Question {i+1}"] = key

    # Save Key
    with open(KEY_FILE, "w") as f:
        json.dump(answer_key, f, indent=2)
        
    print(f"\n✅ BLIND TEST COMPLETE.")
    print(f"👉 Survey: {SURVEY_FILE}")
    print(f"🔑 Key:    {KEY_FILE}")

if __name__ == "__main__":
    main()
