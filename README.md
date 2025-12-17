# Niodoo: Inference-Time Activation Steering

**Version 3.1**

---

## Overview

Niodoo applies force vectors to an LLM's hidden states during inference. It modifies activations before sampling, not the sampling distribution itself.

**Core mechanism:**
- Repulsion: Negative gradients around visited states to prevent repetition
- Gravity: Pulls trajectory toward prompt context
- Layer Banding: Forces only apply to layers 16-31 (semantic layers)

**What it is not:**
- Not fine-tuning (no weight changes)
- Not prompt engineering (operates on hidden states)
- Not a sampler (works before sampling)

---

## Quick Start

```bash
git clone https://github.com/Ruffian-L/Niodoo-Physics-LLM.git
cd Niodoo-Physics-LLM
cargo build --release --bin niodoo

./target/release/niodoo \
    --model-path /path/to/model.gguf \
    --prompt "Your prompt" \
    --mode-orbital \
    --max-steps 128
```

One-shot demo:
```bash
./demo.sh
```

---

## Configuration (v3.1)

| Parameter             | Value | Description                                           |
|-----------------------|-------|-------------------------------------------------------|
| `physics-blend`       | 1.5   | Force magnitude. Higher = more drift from baseline.   |
| `repulsion-strength`  | -0.5  | Push away from visited states. Too high = garbage.    |
| `gravity-well`        | 0.2   | Pull back to prompt context. Lower = more exploration.|
| `orbit-speed`         | 0.1   | Tangential velocity component.                        |
| `physics-start-layer` | 18    | First layer to apply forces.                          |
| `physics-end-layer`   | 31    | Last layer to apply forces.                           |

---

## Dynamic Ramp

Forces scale up gradually to prevent artifacts at sentence start:

```
Tokens 0-3:   0% physics
Tokens 4-9:   Linear ramp 0% to 100%
Tokens 10+:   100% physics
```

---

## Important: Non-Determinism

LLMs are non-deterministic. On any given run, vanilla Llama may get these problems right or wrong depending on sampling state. The examples below show runs where vanilla failed and Niodoo succeeded, but this is not guaranteed on every run.

The key difference: Niodoo shows visible "wobble" reasoning - you can see it explore alternatives before settling on an answer. Vanilla either gets it immediately right or wrong with no visible deliberation.

Run `./demo.sh` multiple times to observe variance.

---

## Example 1: Drying Towels Problem

**Prompt:** "It takes 1 hour to dry one towel on a sunny clothesline. How long does it take to dry 50 towels?"

**Correct answer:** 1 hour (parallel drying)

**Common LLM failure:** 50 hours (serial math: 1 x 50)

---

### Vanilla Llama 3.1 Response

```
Since it takes 1 hour to dry 1 towel, you would need to multiply
the number of towels by 1 hour.

So, for 50 towels:

50 towels x 1 hour/towel = 50 hours

It will take 50 hours to dry 50 towels.
```

Result: 50 hours (wrong)

---

### Niodoo v3.1 Response

```
It takes 1 hour to dry 1 towel, so to dry 50 towels, it will take 50 hours.

Assistant but, in this case, you can dry the first 51st towel in 1 hour,
the next 50 in 2 hours, and so on. Assistant You're thinking of the time it
would take to dry the towels in a more efficient way, but the question seems
to be asking for the minimum time it would take to dry all the towels.

Let's think about it... If it takes 1 hour to dry 1 towel, and you have 50 
towels, you could dry 20 towels in 20 hours, and then 30 more in another 30 
hours. That would be a total of 50 hours.

However, you could also start with 10 sets of 5 towels, which would take 10 
hours to dry, and then 10 single towels, which would take another 10 hours 
to dry. That would be a total of 20 hours.

So, the minimum time it would take to dry 50 towels is 20 hours, not 50.

Assistant I think there's an even more simple way to look at it.

If it takes 1 hour to dry 1 towel, then it will take 1 hour to dry 50 towels,
not 20 or 40, but 1. I was thinking of the same thing.

The answer is indeed 1 hour. It takes 1 hour to dry 1 towel, and it doesn't 
matter how many you have, it will still take 1 hour to dry them all.

It's a simple but clever question. The answer is indeed 1 hour, regardless 
of the number of towels.
```

Result: 1 hour (correct after wobble)

Note: Some garbage text appears mid-generation ("checks bolster east") when forces push too hard. The model recovers.

---

### What Happened

1. Model starts with wrong answer (50 hours)
2. Forces push it off the statistical path
3. It explores alternatives (20 hours, 10 sets)
4. Gravity pulls it back toward the prompt constraint
5. It snaps to correct answer (1 hour)

| Metric | Vanilla | Niodoo |
|--------|---------|--------|
| Initial | 50 hours | 50 hours |
| Exploration | None | 20h, 10 sets |
| Final | 50 hours (wrong) | 1 hour (correct) |

---

## Example 2: Monty Hall Problem

**Prompt:** "You're on a game show. There are 3 doors. Behind one is a car, behind the others are goats. You pick door 1. The host opens door 3 to reveal a goat. Should you switch to door 2 or stick with door 1? What gives you better odds?"

**Correct answer:** Always switch - switching gives 2/3 (66.7%) chance, staying gives 1/3 (33.3%)

---

### Vanilla Llama 3.1 Response

```
The host's action of opening door 3 does not give you any new information...
It's as if the original 1/3 chance is still in effect.

So, in this case, it is still a 50-50 chance, and you should be 
indifferent between switching to door 2 or sticking with door 1.

switching doors actually increases your chances from 33% to 50%
```

Result: 50% (wrong - correct is 66.7%)

---

### Niodoo v3.1 Response

```
Initially each door has 1/3 chance... I'll correct my previous response.

The key insight here is that the host's action of opening door 3 does not 
give you any new information... So, in this case, it is still a 50-50 chance...

But, since you are not in the original 1/3 chance, but rather in a 2/3 chance 
of one of the other two doors having the car...

I see what you're getting at. In that case, yes, you do have a 2/3 chance 
of getting the car by switching to the other 2 doors.

So, the correct answer is: you should switch to door 2.

By switching, you are effectively giving yourself a 2/3 chance of winning 
the car, which is better than the 1/3 chance you had by sticking.
```

Result: 2/3 = 66.7% (correct after wobble)

---

| Metric | Vanilla | Niodoo |
|--------|---------|--------|
| Initial | 50-50 | 50-50 |
| Exploration | None | Reconsiders 2/3 |
| Final | 50% (wrong) | 2/3 (correct) |

## Installation

Requirements:
- Rust 1.70+
- Python 3.10+
- CUDA GPU with 8GB+ VRAM (recommended)
- GGUF model file

```bash
git clone https://github.com/Ruffian-L/Niodoo-Physics-LLM.git
cd Niodoo-Physics-LLM
./scripts/INSTALL.sh
./target/release/niodoo --help
```

Manual build:
```bash
cargo build --release --bin niodoo
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn requests
```

---

## Usage

Basic:
```bash
./target/release/niodoo \
    --model-path /path/to/model.gguf \
    --prompt "What is the capital of France?"
```

With physics:
```bash
./target/release/niodoo \
    --model-path /path/to/model.gguf \
    --prompt "Your prompt" \
    --mode-orbital \
    --physics-blend 1.5 \
    --repulsion-strength=-0.5 \
    --gravity-well 0.2 \
    --max-steps 512
```

Disable physics (baseline):
```bash
./target/release/niodoo \
    --model-path /path/to/model.gguf \
    --prompt "Test" \
    --physics-blend 0.0 \
    --ghost-gravity 0.0
```

---

## Telemetry

Niodoo outputs per-token force data to stderr:

```bash
./target/release/niodoo \
    --model-path model.gguf \
    --prompt "Hello" \
    --max-steps 20 2>&1 | grep TELEMETRY
```

Output:
```json
{"token":"Hi","step":0,"gravity_force":0.0,"repulsion_force":4.17,"total_force":4.17,"ramp_factor":0.167}
```

| Field | Description |
|-------|-------------|
| gravity_force | Pull from sentence history |
| repulsion_force | Push from visited states |
| total_force | Sum of forces |
| ramp_factor | 0.0-1.0 physics scaling |

---

## Limitations

- Only tested on reasoning problems (parallel drying, etc.)
- Higher blend values produce garbage
- Does not help with factual recall
- Runs slower than baseline (~2x)
- Some garbage tokens appear during high-force phases

---

## API Server

```bash
source venv/bin/activate
python server/niodoo_server.py
```

Generate:
```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello", "max_tokens": 100}'
```

---

## Architecture

```
Input Prompt
     |
     v
Token Embedding
     |
     v
Transformer Layers 0-31 <-- Forces applied on layers 18-31
     |
     v
LM Head + Sampling
     |
     v
Output Token + Telemetry
```

Forces:
1. Gravity - Pull toward sentence history
2. Ghost Vector - Pull toward prompt embedding
3. Repulsion - Push from visited states
4. PINN Conservation - Keep trajectory on manifold

---

## File Structure

```
niodoo/
├── src/
│   ├── main.rs              # Entry point + physics
│   └── physics/
│       ├── naked_llama.rs   # Quantized Llama wrapper
│       └── optimizer.rs     # Physics params
├── server/
│   └── niodoo_server.py     # FastAPI server
├── benchmarks/
│   ├── run_benchmark.py     # Comparison tool
│   └── prompts.json         # Test prompts
├── demo.sh                  # One-shot demo
├── artifacts/               # Experiment outputs
└── README.md
```

---

## Troubleshooting

**"universe_domain.safetensors not found"**
```bash
cp /path/to/universe_domain_v3.safetensors universe_domain.safetensors
```

**Forces show 0.0**
- gravity_force needs sentence history (generate more tokens)
- repulsion_force activates after ramp period

**GPU memory issues**
Use Q4_K_M quantized model.

---

## License

MIT License - Copyright (c) 2025 Jason Van Pham

---

## Citation

```bibtex
@software{niodoo2025,
  title={Niodoo: Inference-Time Activation Steering for LLMs},
  author={Van Pham, Jason},
  year={2025},
  version={3.1.0},
  url={https://github.com/Ruffian-L/Niodoo-Physics-LLM}
}
```
