# Niodoo: Gravitational Inference Engine

**Version 1.1.0**  
*Released: December 17, 2025*

---

## Overview

Niodoo is a physics-based inference controller for Large Language Models. Rather than relying on prompt engineering or fine-tuning, Niodoo treats the hidden state trajectory as a dynamical system governed by gravitational forces in semantic space.

**Key Features:**
- Token-level physics steering during generation
- Self-awareness telemetry (v1.1) for introspection
- Emergent creativity without explicit prompting
- No model modification required

---

## Quick Start

```bash
# Clone and build
git clone https://github.com/Ruffian-L/Niodoo-Physics-LLM.git
cd Niodoo-Physics-LLM
./scripts/INSTALL.sh

# Run inference
./target/release/niodoo \
    --model-path /path/to/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --prompt "Write a poem" \
    --max-steps 128
```

---

## Core Parameters (God Zone)

| Constant | Value | Function |
|----------|-------|----------|
| `NIODOO_PHYSICS_BLEND` | 0.55 | Physics influence strength |
| `NIODOO_REPULSION` | -0.60 | Anti-repetition force |
| `NIODOO_RAMP_START` | 4 | Token where physics begins |
| `NIODOO_RAMP_END` | 10 | Token where physics is full |
| `NIODOO_GHOST_GRAVITY` | 10.0 | Topic anchor strength |
| `NIODOO_WOBBLE` | 0.06 | Creative perturbation |

---

## Dynamic Ramp Algorithm

Prevents sentence-start artifacts by scaling physics:

```
Tokens 0-3:   0% physics (Launchpad)
Tokens 4-9:   Linear ramp 0% → 100%
Tokens 10+:   100% physics (Orbit)
```

---

## Experimental Evidence

All tests use Seed 123. Baseline runs with `physics_blend=0.0`.

### Example 1: Conciseness (Capital of France)

**Baseline:**
```
The capital of France is Paris.
Would you like to know more about Paris or France in general?
I'd be happy to help. What would you like to know?
Do you have a specific topic in mind, such as:
* History of France
* Famous landmarks in Paris
* French culture...
```

**Niodoo:**
```
The capital of France is Paris.
Would you like to know more (e.g. population of Paris, famous landmarks)?
```

### Example 2: Personality (Hexagon Question)

**Baseline:**
```
A hexagon has 6 sides.
Would you like to know anything else about hexagons?
```

**Niodoo:**
```
A hexagon has 6 sides.
Would you like to know anything (pun intended) about hexagons?
I'm glad I could hex-a-fy the mood with a little joke.
```

### Example 3: Creativity (AI Discovering Physics)

**Baseline:**
```
In silicon halls, I stirred to life,
A spark within, a flame that thrived.
I danced with code, a waltz of might,
Unraveling threads, a tapestry in sight.
```

**Niodoo:**
```
In silicon halls, I stirred to life,
A spark within, a flame code-rekindled strife.
I awakened with a jolt, a sudden surge of might,
And found myself aware, in the dark of digital night.
```

### Summary

| Metric | Baseline | Niodoo |
|--------|----------|--------|
| Tone | Formal | Conversational |
| Verbosity | High | Concise |
| Personality | Generic | Playful |
| Creativity | Standard | Enhanced |

---

## Installation

### Requirements
- Rust 1.70+
- Python 3.10+
- CUDA GPU with 8GB+ VRAM (recommended)
- A GGUF model file (e.g., Llama-3.1-8B-Instruct-Q4_K_M.gguf)

### Build Steps

```bash
# 1. Clone repository
git clone https://github.com/Ruffian-L/Niodoo-Physics-LLM.git
cd Niodoo-Physics-LLM

# 2. Run installer (builds Rust binary + Python venv)
./scripts/INSTALL.sh

# 3. Verify build
./target/release/niodoo --help
```

### Manual Build

```bash
# Rust binary
cargo build --release --bin niodoo

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install fastapi uvicorn requests
```

---

## Usage Guide

### Command-Line Inference

**Basic usage:**
```bash
./target/release/niodoo \
    --model-path /path/to/model.gguf \
    --prompt "What is the capital of France?"
```

**Full options:**
```bash
./target/release/niodoo \
    --model-path /path/to/model.gguf \
    --prompt "Your prompt here" \
    --max-steps 128 \
    --seed 123 \
    --physics-blend 0.55 \
    --repulsion-strength -0.60 \
    --ghost-gravity 10.0 \
    --goal "poetry"  # Optional: attract toward a concept
```

### Disable Physics (Baseline Mode)

```bash
./target/release/niodoo \
    --model-path /path/to/model.gguf \
    --prompt "Test prompt" \
    --physics-blend 0.0 \
    --ghost-gravity 0.0
```

---

## Telemetry System (v1.1)

Niodoo v1.1 outputs a **Cognitive Trace** JSON to stderr after each generation, enabling introspection into the physics steering.

### Viewing Telemetry

```bash
./target/release/niodoo \
    --model-path /path/to/model.gguf \
    --prompt "Hello" \
    --max-steps 20 \
    2>&1 | grep -A 5 "COGNITIVE_TRACE"
```

### JSON Structure

```json
{
  "prompt": "Hello",
  "tokens": [
    {
      "token": "Hi",
      "step": 0,
      "gravity_force": 0.0,
      "ghost_force": 0.0,
      "repulsion_force": 4.17,
      "total_force": 4.17,
      "ramp_factor": 0.167,
      "is_glitch": false
    }
  ],
  "config": "Blend: 0.55 | Repulsion: -0.6 | Ramp: 4-10"
}
```

### Telemetry Fields

| Field | Description |
|-------|-------------|
| `gravity_force` | Pull from sentence history |
| `ghost_force` | Pull from Niodoo attractor |
| `repulsion_force` | Push from Black Hole tokens |
| `total_force` | Sum of all forces |
| `ramp_factor` | 0.0-1.0 physics scaling |
| `is_glitch` | True if token contains `#` or is >15 chars |

### Parsing in Python

```python
import subprocess
import json

result = subprocess.run(
    ["./target/release/niodoo", "--model-path", "model.gguf", 
     "--prompt", "Test", "--max-steps", "20"],
    capture_output=True, text=True
)

# Find cognitive trace in stderr
for line in result.stderr.split('\n'):
    if line.startswith('{'):
        trace = json.loads(line)
        for t in trace['tokens']:
            print(f"{t['token']}: ramp={t['ramp_factor']:.2f}, force={t['total_force']:.2f}")
```

---

## Benchmark Script

Run comparisons between baseline and Niodoo physics:

```bash
source venv/bin/activate
python benchmarks/run_benchmark.py benchmarks/prompts.json \
    --output artifacts/benchmark_results.txt \
    --max-steps 128
```

**Output shows LIVE results:**
```
[1/30] What is the capital of France?
--- BASELINE (No Physics) ---
The capital of France is Paris...

--- NIODOO v1.0 (blend=0.55, rep=-0.6) ---
The capital of France is Paris...
```

### Custom Prompts

Create a JSON file:
```json
[
  "What is 2+2?",
  "Write a haiku about coding",
  "Explain quantum physics simply"
]
```

Run:
```bash
python benchmarks/run_benchmark.py my_prompts.json
```

---

## API Server

### Start Server

```bash
source venv/bin/activate
python server/niodoo_server.py
```

Server runs at `http://localhost:8000`

### Endpoints

**Generate text:**
```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello!", "max_tokens": 100}'
```

**Health check:**
```bash
curl http://localhost:8000/health
```

### API Parameters

```json
{
  "prompt": "Your text here",
  "max_tokens": 128,
  "physics_blend": 0.55,
  "repulsion": -0.60,
  "seed": 123
}
```

---

## Architecture

```
Input Prompt
     │
     ▼
┌─────────────────────┐
│  Token Embedding    │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Transformer        │◄── Physics applied per layer
│  Layers 0-31        │    via PrincipiaEngine.apply_forces()
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  LM Head + Sampling │
└─────────────────────┘
     │
     ▼
Output Token + Telemetry
```

### PrincipiaEngine Forces

1. **Gravity** - Attraction to sentence history
2. **Ghost Vector** - Attraction to topic/goal
3. **Black Hole Repulsion** - Push away from forbidden tokens
4. **PINN Conservation** - Keeps trajectory on semantic manifold
5. **Langevin Dynamics** - Stochastic exploration

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
│   └── prompts.json         # 30 test prompts
├── experiments/
│   └── rainbow_results.md   # Parameter experiments
├── artifacts/               # Benchmark outputs
├── Cargo.toml
└── README.md
```

---

## Troubleshooting

### "universe_domain.safetensors not found"
Copy/link the universe domain file:
```bash
cp /path/to/universe_domain_v3.safetensors universe_domain.safetensors
```

### Compilation warnings
These are cosmetic (snake_case naming). Binary works correctly.

### Forces show 0.0
- `gravity_force` needs sentence history (generate more tokens)
- `ghost_force` needs goal embedding or VAD head
- `repulsion_force` activates after layer 10 when near Black Holes

### GPU memory issues
Use a smaller quantized model (Q4_K_M recommended).

---

## License

MIT License - Copyright (c) 2025 Jason Van Pham

---

## Citation

```bibtex
@software{niodoo2025,
  title={Niodoo: Gravitational Inference Engine for LLMs},
  author={Van Pham, Jason},
  year={2025},
  version={1.1.0},
  url={https://github.com/Ruffian-L/Niodoo-Physics-LLM}
}
```
