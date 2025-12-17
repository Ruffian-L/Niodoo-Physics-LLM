# Niodoo Phase 2: The Symplectic Engine ("Project Orbit")

We are deleting the "Black Hole" metaphor (which crushes tokens) and replacing it with the **"Satellite" metaphor** (which keeps tokens in stable orbit around the meaning).

## 🛠️ The Phase 2 Checklist

### 1. The Mathematics (The "Jazz")
We need to implement **High-Dimensional Orbital Mechanics**.
Since we are in 4096-dimensional space, we use **Gram-Schmidt Orthogonalization** to find the "Sideways" direction.

**The Equations:**
1. **Gravity (F_g):** The pull towards the "Topic Center" (Context Mean).
2. **Tangential Force (F_tan):** The push that creates the orbit (90 degrees to Gravity).
   *(This subtracts the part of momentum parallel to gravity, leaving only the sideways part).*
3. **Symplectic Update (The Flip):**
   - Momentum += Force * dt
   - Position += Momentum * dt

### 2. Rust Architecture Updates (`src/main.rs`)
We need to add persistent state for the "Motion" of the conversation.

- [ ] **Add State Variable:** `let mut momentum: Vec<f32> = vec![0.0; 4096];` (Initialize to zero).
- [ ] **Add Constants:**
    - `ORBIT_SPEED` (Delta t): Controls how fast we orbit. Start small (`0.1`).
    - `GRAVITY_WELL_STRENGTH`: How hard the topic pulls back.
- [ ] **Update `TokenPhysics` Struct:** Add `momentum_magnitude` to the telemetry.

### 3. The "Bridge" (Latent Space -> Logits)
This is the hardest engineering part. We have a target vector (Position_{Projected}), but we need to pick a *Token*.

**The Optimization Strategy:**
- [ ] **Step A:** Get the Model's Top 50 candidate tokens (standard logits).
- [ ] **Step B:** Extract embeddings for *only* those 50 tokens.
- [ ] **Step C:** Calculate **Cosine Similarity** between each candidate and our Position_{Projected}.
- [ ] **Step D:** Boost the logits of tokens that align with the orbit.

### 4. The Symplectic Loop (Implementation Logic)
Inside the main generation loop:
1. **Calculate Center of Mass:** Average the embeddings of the last 20 tokens (The "Sun").
2. **Calculate Gravity:** Vector from Current Token -> Sun.
3. **Orthogonalize:** Remove the part of `momentum` that points *at* the Sun.
4. **Update Momentum:** Add the orthogonal force.
5. **Project Target:** Where *should* the next token be if it were orbiting?
6. **Steer:** Boost tokens that lie on that path.

### 5. Integration Strategy
- [ ] Add CLI Arg: `--mode orbital` (Default to `newtonian`).
- [ ] Create `fn calculate_orbital_bias()` to encapsulate the heavy math.
- [ ] **Unit Test the Math:** Write a tiny test in Rust that creates two vectors and proves the orthogonalization works.

## 🧪 Benchmarks: How we know it works
| Artifact | Newtonian (v1.0) | Symplectic (v2.0) |
| --- | --- | --- |
| **Logic** | Crashes on "Zebra" (Repulsion) | **Orbits "Zebra"** (Keeps keyword, changes context) |
| **Creativity** | "Lunar Light" (Point Divergence) | **"Celestial Dance"** (Continuous Thematic Flow) |
| **Stability** | Needs "Ramp" to prevent `#ab` | **Inherently Stable** (Orbits don't shatter) |

## 📅 Day 1 Task List
1. **Rest.** (Seriously, v1.1 is huge).
2. **Scaffold:** Create a new branch `feature/orbital-mechanics`.
3. **Math Prototype:** Write a standalone Rust script just to test the `Momentum + Gravity` update logic with dummy vectors.
