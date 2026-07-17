# Niodoo v2.0 Tuning Configuration

**Last Updated:** 2024-12-17

## Golden Configuration (Validated)

| Parameter             | Value | Notes                                                                 |
|-----------------------|-------|-----------------------------------------------------------------------|
| `physics-blend`       | 1.2   | **Reference blend.** Balanced setting for creativity vs. logic.        |
| `repulsion-strength`  | -1.0  | Prevents loops without causing "space debris" artifacts.              |
| `orbit-speed`         | 0.15  | Validated max velocity for 8B models.                                 |
| `gravity-well`        | 1.0   | Standard attraction; higher values crush embedding variance.          |
| `physics-start-layer` | 18    | **CRITICAL.** Layers 0-17 handle syntax; 18-31 handle semantics.      |
| `physics-end-layer`   | 31    | Apply physics through the final layer.           |

### 3. Niodoo v2.1 "Safety Mode" (Default)
The Omni-Tuner results (Dec 2025) identified the optimal balance between Logic (Accuracy) and Noir (Creativity):

| Parameter | Value | Effect |
|---|---|---|
| **Physics Blend** | **0.8**  | High enough for style, low enough to prevent "hallucinated complexity". |
| **Repulsion** | **-0.5** | Gentle nudging away from boring tokens. Higher repulsion (-1.5) breaks logic. |
| **Orbit Speed** | **0.1** | Tight orbit for stability. |
| **Gravity Well** | **0.5** | Reduced from 1.0 to prevent embedding collapse. |
| **Layer Banding** | **16-31** | Apply only to semantic layers (Llama-3 specific). |

**Why this config?**
- **Logic:** "Dry coin" reasoning is preserved (mostly).
- **Style:** "Noir" descriptors (e.g., "lashed down like a dirty secret") are generated.
- **Safety:** "Brake Pedal" ensures clean stops at EOS.

*(Previous "Golden Ratio" of 1.2/-1.0 is deprecated for general use but available for creative-only tasks)*

## Blend Zone Map

| Blend Range | Status | Notes |
|-------------|--------|-------|
| 0.0 - 0.5 | ⚪ Too Weak | Physics has no visible effect |
| 0.5 - 1.0 | 🟢 Stable | Clean output, subtle steering |
| 1.0 - 1.5 | 🟢 Golden Zone | Optimal creativity/stability balance |
| 1.5 - 2.0 | 🟡 Caution | Debris filter recommended |
| 2.0 - 3.0 | 🟠 High Energy | Heavy debris blocking required |
| 3.0+ | 🔴 Experimental | Significant fragmentation |

## Debris Filter Patterns

Tokens blocked when `physics_blend > 1.0`:
```rust
["://", ".Forms", "_REF", "php", ".swing", 
 "http", "www", ".com", "html", "function(", 
 "return ", "var ", "const ", "async ",
 "Angeles", "assistant"]
```

## Telemetry Baselines

| Blend | Avg Force | Glitches | Debris Blocked |
|-------|-----------|----------|----------------|
| 1.0 | ~10 | 0 | 0 |
| 2.0 | ~15 | 0-1 | 20-139 |
| 3.0 | ~20 | 0 | 12-50 |
| 5.0 | ~15 | 0 | 1-8 |

## Test Results by Prompt Category

### At Blend 1.0 (Production-Ready)
- **Logic**: Clean step-by-step reasoning ✅
- **Creative**: Evocative noir prose ("neon signs flicker like fireflies") ✅
- **Ambiguity**: Correctly asks for clarification ✅

### At Blend 3.0+ (Experimental)
- **Logic**: Fragmented but readable
- **Creative**: Needs debris filter for "assistant" leaks
- **Ambiguity**: Heavy filtering required
