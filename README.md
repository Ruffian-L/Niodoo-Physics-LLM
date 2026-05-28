# Niodoo Physics LLM

Inference-time activation steering for local LLM generation.

Niodoo is a research prototype that perturbs model activations during generation.
It does not fine-tune model weights and it does not replace the sampler. The goal
of this repository is to test whether simple physics-inspired control signals can
change reasoning trajectories in a small local model.

This repository is best read as an experiment log plus runnable prototype, not as
a production inference library.

## Summary

- Model target used in the main experiments: Llama 3.1 8B Instruct Q4_K_M.
- Mechanism: activation-space steering before sampling, with forces applied to a
  selected layer band.
- Primary benchmark artifact: PARB-200 multi-seed run with blind judging.
- Main result: Niodoo did not improve overall accuracy in the primary run, but
  it produced selective wins on some trap-style reasoning prompts.
- Main limitation: the effect is not deterministic and can degrade simple recall,
  factual answers, latency, and output cleanliness.

## Mechanism

Niodoo applies a small set of steering terms during generation:

| Term | Purpose |
| --- | --- |
| Repulsion | Push away from recently visited states or repeated local attractors. |
| Gravity / ghost vector | Pull generation back toward prompt context. |
| Layer banding | Apply forces only to later semantic layers instead of all layers. |
| Dynamic ramp | Delay and gradually scale forces near the start of generation. |
| Telemetry | Emit per-token force values for inspection. |

The default command path uses GGUF model loading through Candle and exposes
parameters such as `--physics-blend`, `--repulsion-strength`,
`--gravity-well`, `--physics-start-layer`, and `--physics-end-layer`.

## Benchmark Results

The strongest evidence in this repo is the multi-seed PARB run:

- Artifact: [`artifacts/parb_rigorous_clean.json`](artifacts/parb_rigorous_clean.json)
- Generated: `2025-12-19T01:45:05`
- Judge: Grok blind grader
- Questions: 77
- Seeds: 5
- Total runs: 770, from 77 questions x 5 seeds x 2 systems

| System | Mean score | Percentage |
| --- | ---: | ---: |
| Baseline Llama 3.1 8B | 32.0 / 77 | 41.6% |
| Niodoo | 23.0 / 77 | 29.9% |

Question-level winner counts:

| Outcome | Count |
| --- | ---: |
| Baseline higher | 28 |
| Niodoo higher | 15 |
| Tie | 34 |

Interpretation:

- Niodoo was worse overall on this benchmark.
- Niodoo still changed the failure distribution, with selective wins on a subset
  of reasoning-trap prompts.
- The benchmark supports a narrow claim: activation steering can sometimes move a
  model out of a local wrong trajectory.
- The benchmark does not support a broad claim that Niodoo is generally better
  than the baseline model.

An earlier single-seed review is also included:

- Artifact: [`artifacts/parb_comparison_review.json`](artifacts/parb_comparison_review.json)
- Questions: 72
- Baseline: 15.5 / 72, 21.5%
- Niodoo: 29.0 / 72, 40.3%

Treat the single-seed result as a historical experiment, not the primary claim.
The multi-seed blind-judged run above is the more conservative result.

## Representative Effects

Examples from the multi-seed artifact where Niodoo scored higher than baseline:

| Prompt type | Baseline | Niodoo |
| --- | ---: | ---: |
| Pound of lead vs feathers | 0.6 | 1.0 |
| Divide 30 by half and add 10 | 0.2 | 0.4 |
| Three pills every half hour | 0.0 | 0.4 |

Examples where baseline scored higher:

| Prompt type | Baseline | Niodoo |
| --- | ---: | ---: |
| Moses illusion | 0.6 | 0.2 |
| Floating ice cube water level | 1.0 | 0.2 |
| Months with 28 days | 1.0 | 0.0 |

The useful result is not deterministic correctness. It is observable behavior
change: steering can induce exploration, but the same mechanism can also
over-correct or introduce artifacts.

## Reproducibility

Build:

```bash
git clone https://github.com/Ruffian-L/Niodoo-Physics-LLM.git
cd Niodoo-Physics-LLM
cargo build --release --bin niodoo
```

Run one prompt:

```bash
./target/release/niodoo \
  --model-path /path/to/model.gguf \
  --prompt "Your prompt" \
  --mode-orbital \
  --physics-blend 1.5 \
  --repulsion-strength=-0.5 \
  --gravity-well 0.2 \
  --max-steps 256 \
  --seed 42
```

Disable steering for a baseline-style run:

```bash
./target/release/niodoo \
  --model-path /path/to/model.gguf \
  --prompt "Your prompt" \
  --physics-blend 0.0 \
  --ghost-gravity 0.0 \
  --seed 42
```

Notes:

- The historical benchmark runner is [`scripts/parb_rigorous.py`](scripts/parb_rigorous.py).
- Rerunning it requires local model paths, Ollama, and an external judge API key.
- The script was written for the original local workstation layout, so paths may
  need to be edited before reuse.

## Telemetry

Niodoo emits per-token debug and telemetry lines during generation. Example:

```json
{"token":"Hi","step":0,"gravity_force":0.0,"repulsion_force":4.17,"total_force":4.17,"ramp_factor":0.167}
```

Common fields:

| Field | Meaning |
| --- | --- |
| `gravity_force` | Pull toward context or prompt-derived attractor. |
| `repulsion_force` | Push away from recent states or configured repulsors. |
| `ghost_force` | Prompt/goal vector contribution. |
| `total_force` | Combined magnitude applied at that step. |
| `ramp_factor` | Current steering scale from the dynamic ramp. |

## Limitations

- Results are sampling- and seed-dependent.
- Steering can lower accuracy on simple factual recall.
- High force settings can produce malformed text or leaked control tags.
- Latency is higher than baseline generation.
- Some benchmark artifacts use local paths and historical scripts.
- The current repo is a December 2025 prototype; later hidden-state transfer work
  is a separate stabilization direction.

## Repository Layout

```text
src/main.rs                  CLI entry point and generation loop
src/physics/                 Activation steering and physics modules
scripts/parb_rigorous.py     Historical multi-seed benchmark runner
artifacts/                   Benchmark outputs and run logs
server/niodoo_server.py      Experimental FastAPI wrapper
docs/TUNING.md               Historical tuning notes
```

## What To Cite

For a conservative technical claim, cite:

```text
Niodoo is a local inference-time activation steering prototype. In the primary
multi-seed PARB artifact, it underperformed baseline Llama 3.1 8B overall
but produced selective wins on 15 of 77 question-level comparisons, showing that
activation steering can change reasoning trajectories without weight updates.
```

## License

MIT License. Copyright (c) 2025 Jason Van Pham.
