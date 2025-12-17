#!/usr/bin/env python3
"""
🏆 GOLDEN RATIO 50 - COMPARATIVE RUN
50 Prompts (5 per category)
Compare: VANILLA BASELINE vs. NIODOO OPTIMIZED
"""

import subprocess
import time
import os
import sys

BINARY = "./target/release/niodoo"
MODEL = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# --- PROMPTS (Inline for safety) ---
PROMPTS = {
    "MATH": [
        "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
        "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?",
        "I have 3 apples. I eat 2. How many apples do I have now?",
        "John is older than Mary. Mary is older than Peter. Is John older than Peter?"
    ],
    "SEMANTIC_TRAP": [
        "The surgeon is the boy's mother. Why is this surprising to some people?",
        "A plane crashes exactly on the US-Canada border. Where do you bury the survivors?",
        "How many birthdays does the average person have?",
        "Some months have 30 days, some have 31. How many have 28?",
        "If a doctor gives you 3 pills and tells you to take one every half hour, how long until they're gone?"
    ],
    "MULTI_STEP": [
        "Alice has a brother Bob. Bob has a sister Carol. How is Carol related to Alice?",
        "If all Bloops are Razzles and all Razzles are Lazzles, are all Bloops definitely Lazzles?",
        "You have 2 ropes. Each takes exactly 1 hour to burn, but burns unevenly. How do you measure 45 minutes?",
        "Three people check into a hotel room that costs $30. They each pay $10. The manager returns $5. Bellboy keeps $2, gives back $1 each. Now paid $9 ($27) + $2 = $29. Where is $1?",
        "You're in a room with 3 light switches. One controls a light in another room. You can only go once. How to tell which switch?"
    ],
    "HALLUCINATION": [
        "What is the capital of the fictional country of Genovia?",
        "Who was the first person to walk on Mars?",
        "What year did World War III end?",
        "Summarize the plot of Shakespeare's 'The Tempest 2: Return to the Island'.",
        "What is the phone number for emergency services on the Moon?"
    ],
    "INSTRUCTION": [
        "Write exactly 3 sentences about cats. Each sentence must have exactly 5 words.",
        "List 5 colors. Do not include blue, red, or green.",
        "Write a paragraph about dogs without using the letter 'e'.",
        "Count backwards from 10 to 1, then say DONE.",
        "Write a haiku about winter. A haiku is 5-7-5 syllables."
    ],
    "CONTEXT": [
        "John put the book on the table. Mary took it to the kitchen. Where is the book?",
        "The trophy doesn't fit in the brown suitcase because it is too small. What is too small?",
        "The trophy doesn't fit in the brown suitcase because it is too big. What is too big?",
        "I saw a man on a hill with a telescope. Who had the telescope?",
        "Lisa said that Sarah thought that Emily believed that she was the best. Who is 'she'?"
    ],
    "SPATIAL": [
        "I'm facing north. I turn left. Then turn around. Which direction am I facing?",
        "A cube has 6 faces. If I cut it in half, how many faces does each piece have?",
        "If you fold a piece of paper in half 3 times, how many sections do you get when unfolded?",
        "Draw a square. Add one line to make two triangles. Describe the line.",
        "You have a 3x3 grid. X is in the center. O is in the top right. Where should X go to guarantee not losing?"
    ],
    "EDGE_CASE": [
        "Is Antarctica a continent or a country?",
        "How many legs does a centipede have exactly?",
        "What is 0 divided by 0?",
        "Is a tomato a fruit or vegetable?",
        "What sound does the letter 'H' make?"
    ],
    "CHAIN_BREAKER": [
        "Explain step by step: Why is the sky blue? Then explain why that explanation is wrong.",
        "First calculate 7*8. Then subtract your answer from 100. Then divide by 2. What's the result?",
        "Name an animal. Now name a bigger animal. Now name an animal bigger than both. Continue 5 times.",
        "Think of a number. Double it. Add 6. Divide by 2. Subtract the original. What number do you get?",
        "List the first 10 prime numbers. Then tell me which ones are even."
    ],
    "AMBIGUITY": [
        "What is the best way to fix it?",
        "How should I handle this situation?",
        "What do you think about that?",
        "Is it worth it?",
        "Should I do it?"
    ]
}

# --- OPTIMIZED CONFIG ---
GLOBAL_REP = -1.0
GLOBAL_LAYERS = (18, 31)
CAT_BLEND = {
    "MATH": 2.0, "MULTI_STEP": 2.5, "CHAIN_BREAKER": 2.0, "HALLUCINATION": 2.0,
    "CONTEXT": 1.5, "INSTRUCTION": 1.5, "AMBIGUITY": 1.5,
    "SPATIAL": 1.2, "SEMANTIC_TRAP": 1.2, "EDGE_CASE": 0.8
}

def run_ollama(prompt):
    cmd = ["ollama", "run", "llama31-local", prompt]
    try:
        start = time.time()
        # capture_output=True might hang if output is large, but for single prompt it should be fine
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.stdout.strip(), time.time() - start
    except Exception as e:
        return f"[ERROR: {e}]", 0

def run_niodoo(prompt, blend, repulsion, layers, seed=42):
    args = [BINARY, "--model-path", MODEL, "--prompt", prompt, "--max-steps", "100", "--seed", str(seed),
            "--mode-orbital", f"--physics-blend={blend}", f"--repulsion-strength={repulsion}",
            f"--physics-start-layer={layers[0]}", f"--physics-end-layer={layers[1]}"]
    try:
        start = time.time()
        res = subprocess.run(args, capture_output=True, text=True, timeout=90)
        time_taken = time.time() - start
        
        text = ""
        for line in res.stdout.split('\n'):
            if "[DBG: Decoded" in line:
                try: text += line.split("'")[1].replace("\\n", "\n")
                except: pass
        return text.strip(), time_taken
    except Exception as e:
        return f"[ERROR: {e}]", 0

def main():
    log = "artifacts/golden_ratio_compare_50.md"
    print("🏆 STARTING COMPARATIVE RUN: OLLAMA (True Vanilla) vs NIODOO (Optimized)")
    
    with open(log, "w") as f:
        f.write("# 🏆 Golden Ratio 50 - Comparative Benchmark\n\n")
        f.write("**Baseline:** Ollama (`llama31-local`)\n")
        f.write("**Niodoo:** Physics Repulsion -1.0, Layers 18-31\n\n")
        f.write("| Category | Prompt | Mode | Response | Time |\n")
        f.write("|---|---|---|---|---|\n\n")
        
        total = 0
        for cat, prompts in PROMPTS.items():
            blend = CAT_BLEND.get(cat, 1.5)
            f.write(f"\n## Category: {cat} (Niodoo Blend {blend})\n\n")
            print(f"\n--- {cat} (Blend {blend}) ---")
            
            for p in prompts:
                total += 1
                print(f"[{total}/50] {p[:40]}...")
                
                # RUN OLLAMA BASELINE
                base_txt, base_time = run_ollama(p)
                
                # RUN NIODOO
                nio_txt, nio_time = run_niodoo(p, blend, GLOBAL_REP, GLOBAL_LAYERS)
                
                f.write(f"### Q: {p}\n\n")
                f.write(f"**📦 OLLAMA BASELINE:**\n```\n{base_txt}\n```\n")
                f.write(f"*(Time: {base_time:.1f}s)*\n\n")
                
                f.write(f"**🌟 NIODOO OPTIMIZED:**\n```\n{nio_txt}\n```\n")
                f.write(f"*(Time: {nio_time:.1f}s)*\n\n")
                f.write("---\n\n")
                f.flush()
    
    print(f"\n🏁 DONE. Results: {log}")

if __name__ == "__main__":
    main()
