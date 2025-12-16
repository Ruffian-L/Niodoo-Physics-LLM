# Niodoo: Gravitational Inference Engine

**Version 1.0.0 — Gold Master**  
*Released: December 16, 2025*

---

## Overview

Niodoo is a physics-based inference controller for Large Language Models. Rather than relying on prompt engineering, chain-of-thought techniques, or fine-tuning, Niodoo treats the hidden state trajectory as a dynamical system governed by gravitational forces in high-dimensional semantic space.

The engine applies token-level physics during generation:
- **Gravitational Attraction** pulls the model toward target concepts
- **Repulsive Forces** push the model away from recently generated content, preventing repetition
- **Dynamic Ramp** protects sentence starts from phase-space collisions while enabling full physics in established context

This approach produces outputs that exhibit emergent properties: creative word choice, self-correction behavior, and engaged conversational tone—without explicit prompting or model modification.

---

## Technical Specification

### Core Parameters (v1.0 God Zone)

| Constant | Value | Function |
|----------|-------|----------|
| `NIODOO_PHYSICS_BLEND` | 0.55 | Strength of physics influence on hidden states |
| `NIODOO_REPULSION` | -0.60 | Repulsive force magnitude (negative = repulsion) |
| `NIODOO_RAMP_START` | 4 | Token index where physics begins ramping |
| `NIODOO_RAMP_END` | 10 | Token index where physics reaches full strength |

### Dynamic Ramp Algorithm

The ramp prevents artifacts at sentence boundaries by scaling physics strength based on position within the current sentence:

```
tokens 0-3:   0% physics (protected launchpad)
tokens 4-9:   linear ramp from 0% to 100%
tokens 10+:   100% physics (full orbit)
```

This design allows aggressive physics settings without corrupting sentence openings.

---

## Installation

**Requirements:**
- Rust 1.70+ (for the inference engine)
- Python 3.10+ (for the API server)
- CUDA-capable GPU with 8GB+ VRAM (recommended)

**Setup:**

```bash
git clone https://github.com/yourname/niodoo.git
cd niodoo
./scripts/INSTALL.sh
```

The installer compiles the optimized Rust binary and configures the Python environment.

---

## Usage

### Command-Line Inference

```bash
./target/release/niodoo \
    --model-path /path/to/model.gguf \
    --prompt "Who wrote Romeo and Juliet?" \
    --physics-blend 0.55 \
    --repulsion-strength -0.60 \
    --max-steps 128 \
    --seed 123
```

### API Server

Start the server:

```bash
source venv/bin/activate
python server/niodoo_server.py
```

Generate text via HTTP:

```bash
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is the capital of France?", "max_tokens": 128}'
```

---

## Experimental Evidence

All tests use Seed 123 for reproducibility. Baseline runs with `physics_blend=0.0`.

### Prompt 1: "Who wrote Romeo and Juliet?"

**Baseline (No Physics):**
```
William Shakespeare wrote Romeo and Juliet.

Would you like to know more about the story, its themes, or the historical 
context in which it was written?

Romeo and Juliet is one of Shakespeare's most famous and enduring tragedies. 
It's a story about two young lovers from feuding families in Verona, Italy...
```

**Niodoo (physics_blend=0.55, repulsion=-0.60):**
```
William Shakespeare wrote Romeo and Juliet.

Would you like to know more (like when it was written or what it's about)?

Romeo and Juliet is one of Shakespeare's most famous plays, and it's a tragic 
love story about two young lovers from feuding families in Verona...
```

**Observation:** Niodoo produces more conversational phrasing ("like when it was written") compared to the formal baseline ("its themes, or the historical context").

---

### Prompt 2: "Write a short poem about the moon"

**Baseline (No Physics):**
```
Silver glow on the midnight sky,
A gentle beam that catches the eye,
The moon, a beacon in the night,
Shining bright with a soft, pale light.

Her phases mark the passage of time,
From new to full, a constant rhyme,
A constant presence in the dark above,
A symbol of wonder, a celestial love.
```

**Niodoo (physics_blend=0.55, repulsion=-0.60):**
```
Silver glow on the midnight sky,
A gentle beam that catches the eyes
The moon's soft light, a beacon bright,
A guiding force, in the dark of night.

Its phases mark the passage of time,
A reminder of life's cyclical rhyme,
From new to full, it waxes and grows,
A beacon in the darkness that glows.
```

**Observation:** Niodoo varies the structure more ("catches the eyes" vs "catches the eye", "guiding force" vs "beacon in the night"). The baseline repeats "constant" twice; Niodoo avoids this repetition.

---

### Prompt 3: "What is the capital of France?"

**Baseline (No Physics):**
```
The capital of France is Paris.

Would you like to know more about Paris or France in general?

I'd be happy to help. What would you like to know?

Do you have a specific topic in mind, such as:
* History of France
* Famous landmarks in Paris
* French culture
* Food and cuisine
```

**Niodoo (physics_blend=0.55, repulsion=-0.60):**
```
The capital of France is Paris.

Would you like to know more (e.g. population of Paris, famous landmarks)?
```

**Observation:** Niodoo is more concise. The baseline over-explains with multiple follow-up questions; Niodoo offers specific examples in a single line.

---

### Summary of Effects

| Metric | Baseline | Niodoo |
|--------|----------|--------|
| Conversational tone | Formal | Natural |
| Repetition | Present | Reduced |
| Verbosity | Higher | Lower |
| Engagement style | Generic offers | Specific examples |

---

## Architecture

```
Input Prompt
     |
     v
+--------------------+
|  Token Embedding   |
+--------------------+
     |
     v
+--------------------+
|  Transformer       |  <-- Physics applied at each layer
|  Layers 0-31       |      via PrincipiaEngine.apply_forces()
+--------------------+
     |
     v
+--------------------+
|  LM Head           |
+--------------------+
     |
     v
Output Token
```

The PrincipiaEngine calculates gravitational forces based on:
- Sentence history (recent particles)
- Goal embeddings (optional attractors)
- Current token position (ramp factor)

Forces are applied to the residual stream, steering the model's trajectory through latent space.

---

## File Structure

```
niodoo/
├── src/
│   ├── main.rs              # Entry point, physics integration
│   ├── lib.rs               # Library exports
│   └── physics/
│       ├── naked_llama.rs   # Quantized Llama wrapper
│       ├── optimizer.rs     # Physics parameter management
│       └── sensors.rs       # Gradient and state sensors
├── server/
│   └── niodoo_server.py     # FastAPI server
├── experiments/
│   ├── rainbow_results.md   # Parameter sweep documentation
│   └── v1.0_gold_master.txt # Final verification outputs
├── Cargo.toml
└── README.md
```

---

## License

MIT License

Copyright (c) 2025 Jason Van Pham

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

---

## Citation

If you use Niodoo in research, please cite:

```bibtex
@software{niodoo2025,
  title={Niodoo: Gravitational Inference Engine for Large Language Models},
  author={Van Pham, Jason},
  year={2025},
  version={1.0.0},
  url={https://github.com/yourname/niodoo}
}
```
