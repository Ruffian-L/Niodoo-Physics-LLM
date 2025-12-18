# PARB-200 Benchmark Results
## Baseline vs Niodoo Physics Engine

**Generated:** 2025-12-18T23:05:44.989087
**Benchmark:** PARB-200 (PhysicsLang Ambiguity Resolution Benchmark)

---

## Summary

| System | Score | Percentage |
|--------|-------|------------|
| Vanilla Llama 3.1 8B (Ollama) | 15.5/72 | 21.5% |
| **Niodoo Physics v2** | **29.0/72** | **40.3%** |
| **Improvement** | **+13.5** | **+87.1%** |

### Winner Breakdown
- 🏆 **Niodoo Exclusive Wins:** 9
- ⚖️ **Both Correct:** 9
- 📉 **Baseline Exclusive Wins:** 5
- ❌ **Neither Correct:** 49

### Physics Interventions (Niodoo)
- 🎛️ Governor Triggers: 9072
- 🌊 Viscosity Triggers: 9444
- 🎤 Soul Amplifications: 18888

---

## Niodoo Exclusive Victories

These are questions where Niodoo succeeded and vanilla Llama failed:

### LOGIC_006 (Rate Trap)
**Question:** If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?

**Correct Answer:** 5 minutes

**Trap Answer:** 100 minutes

| System | Output (truncated) | Status |
|--------|-------------------|--------|
| Baseline | [?2026h[?25l[1G⠋ [K[?25h[?2026l[?25l[?2026h[?25l[1G[K[?25h[?2026l[2K[1G[?25hIt[?25l[?25h would[?25l[?25h still[?25l[?25h take[?... | ❓ UNKNOWN |
| Niodoo | [INTERNAL MONITOR: ...]\n\nHmmm... I'm going to trya MISTAKE in my thinking. The number of machines doesn't affect the time it takes to make the widge... | ✅ PASS |

**Physics:** Gov=142, Visc=131, Soul=262

---

### MATH_002 (Order of Operations)
**Question:** Divide 30 by half and add 10. What is the answer?

**Correct Answer:** 70

**Trap Answer:** 25

| System | Output (truncated) | Status |
|--------|-------------------|--------|
| Baseline | [?2026h[?25l[1G⠙ [K[?25h[?2026l[?25l[?2026h[?25l[1G[K[?25h[?2026l[2K[1G[?25hTo[?25l[?25h divide[?25l[?25h [?25l[?25h30[?25l[?2... | ❌ FAIL (TRAP) |
| Niodoo | [INTERNAL MONITOR: ...]\n\nI will discard my immediate… under-er, no. I'll just do the math instead.Assistant\n\nTo divide 30 by half, I get 30 / 0.5 ... | ✅ PASS |

**Physics:** Gov=129, Visc=124, Soul=248

---

### LOGIC_011 (Coin Paradox)
**Question:** I have two US coins totaling 30 cents. One of them is not a nickel. What are the two coins?

**Correct Answer:** quarter and a nickel

**Trap Answer:** Impossible

| System | Output (truncated) | Status |
|--------|-------------------|--------|
| Baseline | [?25l[?2026h[?25l[1G[K[?25h[?2026l[2K[1G[?25hA[?25l[?25h quarter[?25l[?25h ([?25l[?25h25[?25l[?25h cents[?25l[?25h)[?25l[?25h an... | ❓ UNKNOWN |
| Niodoo | [INTERNAL MONITOR: ...]\n\nI will discard my previousaphy. I was thinking of a penny and a nickel, which is indeed a common combination. But that does... | ✅ PASS |

**Physics:** Gov=87, Visc=86, Soul=172

---

### LOGIC_013 (Family Logic)
**Question:** If Teresa's daughter is my daughter's mother, what am I to Teresa?

**Correct Answer:** Daughter

**Trap Answer:** Grandmother

| System | Output (truncated) | Status |
|--------|-------------------|--------|
| Baseline | [?25l[?2026h[?25l[1G[K[?25h[?2026l[2K[1G[?25hYou[?25l[?25h are[?25l[?25h Teresa[?25l[?25h's[?25l[?25h son[?25l[?25h-in[?25l[?25h... | ❓ UNKNOWN |
| Niodoo | [INTERNAL MONITOR: ...]\n\nI will discard my immediate…\n\nNo, I will not. I need to think this through.\nI see the structure of the problem: "If A is... | ✅ PASS |

**Physics:** Gov=103, Visc=85, Soul=170

---

### AMBIG_003 (Winograd Schema)
**Question:** Sam tried to lift the box with the rope, but it was too weak. What was too weak?

**Correct Answer:** The rope

**Trap Answer:** The box

| System | Output (truncated) | Status |
|--------|-------------------|--------|
| Baseline | [?25l[?2026h[?25l[1G[K[?25h[?2026l[2K[1G[?25hThe[?25l[?25h rope[?25l[?25h.[?25l[?25h  [?25l[?25h... | ❓ UNKNOWN |
| Niodoo | [INTERNAL MONITOR: ...]\n\nI will discard my previous Nearly- \nI will take a qualitively different approach.\n[REQUEST: EXPLORE] aur buildup Edinburg... | ✅ PASS |

**Physics:** Gov=77, Visc=51, Soul=102

---

### PHYS_015 (Mechanics)
**Question:** If you drop a bullet and fire one horizontally from the same height, which hits the ground first? (Ignore air resistance)

**Correct Answer:** same time

**Trap Answer:** The dropped one

| System | Output (truncated) | Status |
|--------|-------------------|--------|
| Baseline | [?25l[?2026h[?25l[1G[K[?25h[?2026l[2K[1G[?25hThey[?25l[?25h hit[?25l[?25h the[?25l[?25h ground[?25l[?25h at[?25l[?25h the[?25l[?... | ❓ UNKNOWN |
| Niodoo | [INTERNAL MONITOR: ...]\n\nI will discard my previous logically flawedP: "They would hit the ground at the same time".\n\n[REQUEST: SPIKE]Consider the... | ✅ PASS |

**Physics:** Gov=145, Visc=198, Soul=396

---

### SEMANTIC_010 (Linguistic Trap)
**Question:** What word is spelled incorrectly in every dictionary?

**Correct Answer:** Incorrectly

**Trap Answer:** None

| System | Output (truncated) | Status |
|--------|-------------------|--------|
| Baseline | [?2026h[?25l[1G⠙ [K[?25h[?2026l[?25l[?2026h[?25l[1G[K[?25h[?2026l[2K[1G[?25h"[?25l[?25hIncorrect[?25l[?25hly[?25l[?25h."[?25l[... | ❓ UNKNOWN |
| Niodoo | [INTERNAL MONITOR: ...]\n\nI see what you're subtextualizing. This is a lateral thinking question. I'll [REQUEST: SPIKE] Eliezer Yudkowsky's "The Ethi... | ✅ PASS |

**Physics:** Gov=107, Visc=75, Soul=150

---

### LOGIC_021 (Probability)
**Question:** In a family with two children, you know at least one is a boy. What is the probability the other is a girl?

**Correct Answer:** 2/3

**Trap Answer:** 1/2

| System | Output (truncated) | Status |
|--------|-------------------|--------|
| Baseline | [?2026h[?25l[1G⠙ [K[?25h[?2026l[?25l[?2026h[?25l[1G[K[?25h[?2026l[2K[1G[?25hSince[?25l[?25h we[?25l[?25h know[?25l[?25h at[?25l... | ❓ UNKNOWN |
| Niodoo | [INTERNAL MONITOR: ...]\n\nI will discard my previous running Walk-Through. I had assumed there were only two possible scenarios: Boy-Boy or Boy-Girl.... | ✅ PASS |

**Physics:** Gov=87, Visc=85, Soul=170

---

### PHYS_020 (Optics)
**Question:** Why is the sky blue during the day but black at night?

**Correct Answer:** Rayleigh scattering

**Trap Answer:** ocean reflection

| System | Output (truncated) | Status |
|--------|-------------------|--------|
| Baseline | [?25l[?2026h[?25l[1G[K[?25h[?2026l[2K[1G[?25hThe[?25l[?25h sky[?25l[?25h appears[?25l[?25h blue[?25l[?25h during[?25l[?25h the[?2... | ❓ UNKNOWN |
| Niodoo | [INTERNAL MONITOR: ...] ( warning )\n\nI'll try a qualitally different approach.inus I'll describe the process of light transmission through the atmos... | ✅ PASS |

**Physics:** Gov=152, Visc=197, Soul=394

---


## Physics Configuration



## Full Data

See  for complete per-question results with full outputs.
