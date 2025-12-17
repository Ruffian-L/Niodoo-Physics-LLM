# Niodoo v1.1 Release Notes ("The Mirror")
**Date:** December 17, 2025

## 🚀 New Features

### 1. Autonomic Regulation ("The Heartbeat")
- **Dynamic Physics Blending:** The system now monitors "stress" (high force/glitches) and "boredom" (low orbit variance).
- **Homeostasis Loop:** Automatically adjusts `physics_blend` and `repulsion` between generation turns to maintain optimal creativity without breakdown.
- **Python Integration:** New `AutonomicRegulator` class in `niodoo_server.py`.

### 2. Recursive Self-Model ("The Mirror")
- **Telemetry Injection:** The server captures the raw physics telemetry (gravity, momentum, repulsion) from the Rust engine.
- **Self-Explanation:** When asked "Why?", the model receives a system prompt containing its own previous internal state (e.g., "You felt a repulsion force of -5.0").
- **Meta-Cognition:** Enables the model to ground its creative choices in its physical constraints (Experimental).

### 3. Orbital Mechanics Update
- **"Sun Anchor" Fix:** Gravity now activates immediately (`len >= 1`) to prevent initial drift.
- **Live Centroid Tracking:** The "Sun" (context center) is now calculated using live sentence embeddings rather than just completed history.

---

## 🛑 Known Limitations (Red Team Report)

Based on rigorous Red Teaming (Dec 17, 2025):

1. **Repulsion Resistance:** The underlying Llama-3 model has strong priors for repetition in certain contexts (e.g., repeating specific words). It often resists Niodoo's repulsion forces even when they are physically high.
2. **Gravity Telemetry Gap:** While orbital mechanics are active, the specific `gravity_force` telemetry field currently reports `0.0`. This is a measurement/reporting bug, though the orbital steering itself is functional.
3. **Mirror Blindness:** The context injection for "The Mirror" works technically, but the model often ignores the injected physics data in favor of generic "helpful assistant" responses due to RLHF training.

---

## 📦 Upgrade Instructions

1. **Update Code:** `git pull origin main`
2. **Rebuild Binary:** `./scripts/INSTALL.sh` (or `cargo build --release --bin niodoo`)
3. **Update Python Deps:** `pip install fastapi uvicorn pydantic requests`
