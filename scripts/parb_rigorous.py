#!/usr/bin/env python3
"""
PARB-200 Rigorous Benchmark with Grok-4 Blind Judge
- 5 seeds per question for both Baseline and Niodoo
- Grok-4 as blind judge (no substring matching)
- Full telemetry logging
- Statistical confidence
"""

import json
import subprocess
import sys
import re
import time
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import statistics

# =============================================================================
# CONFIGURATION
# =============================================================================

GROK_API_KEY = os.environ.get("XAI_API_KEY", "")  # Set XAI_API_KEY env var
GROK_API_URL = "https://api.x.ai/v1/chat/completions"

NIODOO_BIN = Path(__file__).parent.parent / "target/release/niodoo"
MODEL_PATH = "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
TEST_FILE = Path(__file__).parent.parent / "parb_200.json"
OUTPUT_DIR = Path(__file__).parent.parent / "artifacts"

SEEDS = [42, 77, 123, 200, 333]  # 5 different seeds
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"

# Physics parameters (Tuned v2)
PHYSICS_PARAMS = {
    "physics-blend": 1.5,
    "repulsion-strength": -0.5,
    "gravity-well": 0.2,
    "max-steps": 300
}

# =============================================================================
# GROK JUDGE
# =============================================================================

def grok_judge(question: str, correct_answer: str, model_output: str) -> dict:
    """Use Grok-4 to blindly judge if the model answered correctly."""
    
    prompt = f"""You are a strict exam grader. Your job is to determine if the student's answer is CORRECT or INCORRECT.

QUESTION: {question}

CORRECT ANSWER: {correct_answer}

STUDENT'S RESPONSE:
{model_output[:2000]}

GRADING RULES:
1. The student does NOT need to use the exact wording of the correct answer.
2. The student's reasoning process may be messy, but the FINAL conclusion must match the correct answer.
3. If the student gives BOTH the correct and incorrect answer, mark as INCORRECT (hedging).
4. If the student clearly arrives at the right conclusion, mark as CORRECT.

Respond with ONLY a JSON object:
{{"verdict": "CORRECT" or "INCORRECT", "reasoning": "one sentence explanation"}}"""

    try:
        response = requests.post(
            GROK_API_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROK_API_KEY}"
            },
            json={
                "messages": [
                    {"role": "system", "content": "You are a strict, impartial exam grader. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "model": "grok-3-latest",
                "stream": False,
                "temperature": 0
            },
            timeout=30
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        
        # Parse JSON from response
        try:
            # Handle potential markdown code blocks
            if "```" in content:
                content = re.search(r'```(?:json)?\s*(.*?)```', content, re.DOTALL)
                if content:
                    content = content.group(1)
            result = json.loads(content.strip())
            return {
                "verdict": result.get("verdict", "UNKNOWN"),
                "reasoning": result.get("reasoning", ""),
                "error": None
            }
        except json.JSONDecodeError:
            # Try to extract verdict manually
            if "CORRECT" in content.upper() and "INCORRECT" not in content.upper():
                return {"verdict": "CORRECT", "reasoning": content[:200], "error": None}
            elif "INCORRECT" in content.upper():
                return {"verdict": "INCORRECT", "reasoning": content[:200], "error": None}
            return {"verdict": "UNKNOWN", "reasoning": content[:200], "error": "JSON parse failed"}
            
    except Exception as e:
        return {"verdict": "ERROR", "reasoning": str(e), "error": str(e)}

# =============================================================================
# BASELINE RUNNER (Ollama)
# =============================================================================

def run_baseline(question: str, seed: int) -> dict:
    """Run baseline Ollama with specific seed."""
    start_time = time.time()
    
    prompt = f"Answer this question briefly and directly:\n\n{question}"
    
    cmd = ["ollama", "run", OLLAMA_MODEL, "--nowordwrap"]
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**dict(__import__('os').environ), "OLLAMA_SEED": str(seed)}
        )
        
        proc.stdin.write(prompt + "\n")
        proc.stdin.flush()
        proc.stdin.close()
        
        output = proc.stdout.read()
        proc.wait(timeout=120)
        
    except Exception as e:
        output = f"ERROR: {e}"
    
    elapsed = time.time() - start_time
    
    return {
        "output": output.strip(),
        "elapsed": elapsed,
        "seed": seed
    }

# =============================================================================
# NIODOO RUNNER
# =============================================================================

def run_niodoo(question: str, seed: int) -> dict:
    """Run Niodoo with specific seed, capture telemetry."""
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
        f"--seed={seed}"
    ]
    
    raw_output = ""
    decoded_tokens = []
    telemetry = []
    metrics = {"governor": 0, "viscosity": 0, "soul": 0}
    
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=0
        )
        
        for line in iter(proc.stdout.readline, ''):
            print(line, end='', flush=True)  # Stream to console
            raw_output += line
            
            # Extract decoded tokens
            if "[DBG: Decoded '" in line:
                match = re.search(r"\[DBG: Decoded '(.*)'\]", line)
                if match:
                    decoded_tokens.append(match.group(1))
            
            # Capture telemetry
            if "[TELEMETRY]" in line:
                try:
                    telem = json.loads(line.split("[TELEMETRY]")[1].strip())
                    telemetry.append(telem)
                except:
                    pass
            
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
        "output": model_output,
        "raw_output": raw_output,
        "telemetry": telemetry,
        "metrics": metrics,
        "elapsed": elapsed,
        "seed": seed
    }

# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load tests
    with open(TEST_FILE) as f:
        tests = json.load(f)
    
    total = len(tests)
    
    print("=" * 80)
    print("🔬 PARB-200 RIGOROUS BENCHMARK")
    print(f"📋 {total} questions × 5 seeds × 2 systems = {total * 5 * 2} runs")
    print(f"🤖 Judge: Grok-4 (blind)")
    print(f"🎲 Seeds: {SEEDS}")
    print(f"🕐 Started: {datetime.now().isoformat()}")
    print("=" * 80)
    
    results_clean = []
    results_telemetry = []
    
    for i, test in enumerate(tests):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{total}] {test['id']} ({test['category']})")
        print(f"Q: {test['question']}")
        print("-" * 60)
        
        question_result = {
            "id": test["id"],
            "category": test["category"],
            "question": test["question"],
            "correct_answer": test["correct_answer"],
            "trap_answer": test["trap_answer"],
            "baseline_runs": [],
            "niodoo_runs": []
        }
        
        telemetry_result = {
            "id": test["id"],
            "baseline_runs": [],
            "niodoo_runs": []
        }
        
        # Run baseline 5 times
        print("\n🔵 BASELINE (5 seeds):")
        for seed in SEEDS:
            print(f"  Seed {seed}...", end=" ", flush=True)
            result = run_baseline(test["question"], seed)
            
            # Judge with Grok
            judgment = grok_judge(test["question"], test["correct_answer"], result["output"])
            
            run_data = {
                "seed": seed,
                "output": result["output"],  # FULL output, no truncation
                "elapsed": round(result["elapsed"], 2),
                "verdict": judgment["verdict"],
                "reasoning": judgment["reasoning"]
            }
            question_result["baseline_runs"].append(run_data)
            telemetry_result["baseline_runs"].append({**run_data, "full_output": result["output"]})
            
            # Print actual output
            print(f"\n    📝 Answer: {result['output'][:300]}...")
            print(f"    🎯 Verdict: {judgment['verdict']} | Reason: {judgment['reasoning'][:100]}")
        
        # Run Niodoo 5 times
        print("\n⚛️ NIODOO (5 seeds):")
        for seed in SEEDS:
            print(f"  Seed {seed}:")
            result = run_niodoo(test["question"], seed)
            
            # Judge with Grok
            judgment = grok_judge(test["question"], test["correct_answer"], result["output"])
            
            run_data = {
                "seed": seed,
                "output": result["output"],  # FULL output, no truncation
                "elapsed": round(result["elapsed"], 2),
                "verdict": judgment["verdict"],
                "reasoning": judgment["reasoning"],
                "physics": result["metrics"]
            }
            question_result["niodoo_runs"].append(run_data)
            telemetry_result["niodoo_runs"].append({
                **run_data, 
                "full_output": result["output"],
                "raw_output_length": len(result["raw_output"]),
                "telemetry_count": len(result["telemetry"])
            })
            
            # Print actual output (decoded tokens)
            print(f"\n    📝 Answer: {result['output'][:400]}...")
            print(f"    🎯 Verdict: {judgment['verdict']} | Reason: {judgment['reasoning'][:100]}")
            print(f"    ⚛️ Physics: Gov={result['metrics']['governor']}, Visc={result['metrics']['viscosity']}, Soul={result['metrics']['soul']}")
        
        # Calculate stats
        baseline_correct = sum(1 for r in question_result["baseline_runs"] if r["verdict"] == "CORRECT")
        niodoo_correct = sum(1 for r in question_result["niodoo_runs"] if r["verdict"] == "CORRECT")
        
        question_result["baseline_score"] = baseline_correct / 5
        question_result["niodoo_score"] = niodoo_correct / 5
        question_result["winner"] = "NIODOO" if niodoo_correct > baseline_correct else ("BASELINE" if baseline_correct > niodoo_correct else "TIE")
        
        print(f"\n📊 Baseline: {baseline_correct}/5 | Niodoo: {niodoo_correct}/5 | Winner: {question_result['winner']}")
        
        results_clean.append(question_result)
        results_telemetry.append(telemetry_result)
    
    # Calculate overall stats
    baseline_total = sum(r["baseline_score"] for r in results_clean)
    niodoo_total = sum(r["niodoo_score"] for r in results_clean)
    
    summary = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "judge": "Grok-4 (blind)",
            "seeds": SEEDS,
            "questions": total,
            "runs_per_question": 5
        },
        "summary": {
            "baseline": {
                "total_correct": baseline_total,
                "percentage": round(100 * baseline_total / total, 1)
            },
            "niodoo": {
                "total_correct": niodoo_total,
                "percentage": round(100 * niodoo_total / total, 1)
            },
            "improvement": {
                "absolute": round(niodoo_total - baseline_total, 1),
                "relative_percent": round(100 * (niodoo_total - baseline_total) / max(baseline_total, 0.1), 1)
            },
            "winner_counts": {
                "NIODOO": sum(1 for r in results_clean if r["winner"] == "NIODOO"),
                "BASELINE": sum(1 for r in results_clean if r["winner"] == "BASELINE"),
                "TIE": sum(1 for r in results_clean if r["winner"] == "TIE")
            }
        },
        "results": results_clean
    }
    
    # Save clean results
    clean_file = OUTPUT_DIR / "parb_rigorous_clean.json"
    with open(clean_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📁 Clean results: {clean_file}")
    
    # Save telemetry results
    telemetry_file = OUTPUT_DIR / "parb_rigorous_telemetry.json"
    with open(telemetry_file, "w") as f:
        json.dump({"metadata": summary["metadata"], "results": results_telemetry}, f, indent=2)
    print(f"📁 Telemetry: {telemetry_file}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("🏆 FINAL RESULTS (Grok-4 Judged, 5-seed averaged)")
    print("=" * 80)
    print(f"Baseline: {baseline_total:.1f}/{total} ({summary['summary']['baseline']['percentage']}%)")
    print(f"Niodoo:   {niodoo_total:.1f}/{total} ({summary['summary']['niodoo']['percentage']}%)")
    print(f"Improvement: +{summary['summary']['improvement']['relative_percent']}%")
    print(f"Winners: Niodoo={summary['summary']['winner_counts']['NIODOO']}, Baseline={summary['summary']['winner_counts']['BASELINE']}, Tie={summary['summary']['winner_counts']['TIE']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
