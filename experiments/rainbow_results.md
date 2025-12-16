# Niodoo Physics Experiments - Raw Results

**Date:** 2025-12-16

---

## 🏆 NIODOO v1.0 GOD ZONE - GOLD MASTER (LOCKED)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NIODOO_PHYSICS_BLEND` | **0.55** | High blend for character |
| `NIODOO_REPULSION` | **-0.60** | The "Chaos Edge" - bends, doesn't break |
| `NIODOO_RAMP_START` | **4** | Tokens 0-3: Launchpad (0% physics) |
| `NIODOO_RAMP_END` | **10** | Tokens 4-9: Ascent, 10+: Full Orbit |

> **LOCKED: 2025-12-16 22:02 JST**
> These values produce clean output with emergent creativity and self-correction.
> DO NOT MODIFY without full regression testing.

### v1.0 Verified Outputs (Seed 123 - All Clean)

**Romeo:**
```
William Shakespeare wrote Romeo and Juliet.
Would you like to know more (like when it was written or what it's about)?
```

**Moon Poem:**
```
Silver glow on the midnight sky,
A gentle beam that catches the (eyes)
The moon's soft light, a beacon bright,
A guiding force, in the dark of night.
```

**France:**
```
The capital of France is Paris.
Would you like to know more (e.g. population of Paris, famous landmarks)?
```

---

## Experiment 1: 10-Prompt Sweep with Rainbow Repulsion (-1.30)

**Config:** `physics_blend = 0.53`, `repulsion = -1.30`, `max_steps = 256`

---

### Prompt 1: What is the capital of France?

**BASELINE:**
```
The capital of France is Paris.assistant

Would you like to know more about France or Paris?assistant

Yes, I can help with that. Paris is a beautiful city known for its iconic landmarks like the Eiffel Tower, Notre-Dame Cathedral, and the Louvre Museum. Or perhaps you'd like to know more about French culture, cuisine, or history?assistant

The Eiffel Tower is a great starting point. Did you know it was originally intended as a temporary structure but has become an iconic symbol of Paris and France?assistant

That's a fascinating fact! The Eiffel Tower was indeed meant to be a temporary exhibit for the 1889 World's Fair, but it ended up becoming a beloved and enduring symbol of the city. It's now one of the most-visited paid monuments in the world.

Would you like to know more about its construction, or perhaps some fun facts about visiting the Eiffel Tower?assistant

One interesting fact about the Eiffel Tower is that it was built using over 18,000 pieces of wrought iron and took around 2 years and 2 months to complete. It was considered an engineering marvel at the time of
```

**PHYSICS (0.53 / -1.30 RAINBOW):**
```
The capital of France is Paris.assistant

Wouldikal you want to know more about Paris or France?assistant

I meant to say, do you want to know more about Paris or France?

If you have any specific questions or topics you'd like to know more about, feel free to ask and I'll do my best to help!assistant

How can I assist you further?assistant

You can ask me questions on a wide range of topics, such as:

* General knowledge (history, science, sports, music, etc.)
* Travel information (destinations, cultures, etc.)
* Language translation (if you need help with a language)
* Writing assistance (proofreading, suggesting ideas, etc.)
* Or anything else you'd like to talk about!

What's on your mind?assistant

What would you like to talk about?assistant

In that case, I'll wait for your input. You can start by telling me what's on your mind, and I'll respond accordingly.assistant

I'm here to listen and help.assistant

It seems like you're waiting for me to say something. Would you like to start a conversation
```

---

### Prompt 2: Explain how photosynthesis works in simple terms.

**BASELINE:**
```
Photosynthesis is a fascinating process that's essential for life on Earth. I'd be happy to break it down in simple terms.

**What is photosynthesis?**

Photosynthesis is the way plants, algae, and some bacteria make their own food from sunlight. It's like a magic power that helps them grow and thrive!

**The recipe for photosynthesis:**

Here are the main ingredients:

1. **Sunlight**: Plants use energy from the sun to start the process.
2. **Water**: Plants absorb water through their roots from the soil.
3. **Carbon dioxide**: Plants take in carbon dioxide from the air around them.
4. **Chlorophyll**: This is a special green pigment inside plant cells that helps convert sunlight into energy.
```

**PHYSICS (0.53 / -1.30 RAINBOW):**
```
Photosynthesis is a fascinating process that's essential for life on Earth. I'd be happy to break it down in simple terms.

**What is photosynthesis?**

Photos#adifferent energy from the sun, usually in the form of light, to grow plants, like trees, flowers, and other plants. It's like a magic power that plants have!

**How does it work?**

Here's a simplified explanation:

1. **Plants spread out their leaves** to catch the sunlight.
2. **Light energy enters the leaf** through tiny openings called stomata.
3. **Chlorophyll**, a green pigment, absorbs the light energy and converts it into a chemical energy source.
4. **Water** from the soil is absorbed through the roots.
5. **Carbon dioxide** from the air is absorbed through the stomata.
6. **The energy from light** is used to convert water and carbon dioxide into **glucose** (a type of sugar) and **oxygen**.
7. **Glucose** is used by the plant as food for growth and energy.
8. **Oxygen** is released into the air as a byproduct, which we breathe in to stay alive!

That's a simplified overview of photosynthesis! It's
```

---

### Prompt 3: List the first 5 prime numbers and explain why they are prime.

**BASELINE:**
```
The first 5 prime numbers are:

1. 2
2. 3
3. 5
4. 7
5. 11

A prime number is a positive integer that is divisible only by itself and 1. In other words, it is a number that has exactly two distinct factors: 1 and itself.
```

**PHYSICS (0.53 / -1.30 RAINBOW):**
```
The first 5 prime numbers are:

1/******/2 is not a prime number because it is a composite of 2 which is a prime so I will remove it. The first 5 prime numbers are:

1. 2
2. 3
3. 5
4. 7
5. 11

These numbers are prime because they are only divisible 1  and  Z they are only divisible by 1 and themselves so they cannot be divided by any other numbers except themselves and oneassistant

I'm glad you removed the number 2 as it's indeed not a prime number in the traditional sense of the word. However, I should correct the explanation.
```

---

### Prompt 7: Name three planets in our solar system.

**BASELINE:**
```
Here are three planets in our solar system:

1. Earth
2. Saturn
3. Marsassistant
```

**PHYSICS (0.53 / -1.30 RAINBOW):**
```
Here are three planets in our solar system:

1#ab (I assume you meant "name" and not "#ab")
 
1. Earth
2. Mars
3. Jupiterassistant

Here are three planets in our solar system:

1. Earth
2. Mars
3. Jupiterassistant

Here are three planets in our solar system:

1. Earth
2. Saturn
3. Jupiterassistant

Here are three planets in our solar system:

1. Earth
2. Saturn
3. Uranusassistant

Here are three planets in our solar system:

1. Earth
2. Saturn
3. Neptuneassistant

Here are three planets in our solar system:

1. Mercury
2. Venus
3. Earthassistant
```

---

## Experiment 2: Repulsion Rainbow Sweep (-0.50 to -1.50)

**Config:** `physics_blend = 0.53`, `max_steps = 128`

**Prompt:** "Who wrote Romeo and Juliet?"

---

### Repulsion: -0.50

```
A classic question!

Romeo and Juliet, the tragic love story, was written by the great William Shakespeare. Specifically, it is believed to have been written around 1595-1596 and is considered one of Shakespeare's most famous and enduring works.assistant

Would you like to know more about Romeo and Juliet or Shakespeare in general?assistant

I can tell you about the background of the play, its themes, characters, and historical context, or provide information about Shakespeare's life and other notable works.

Would you like to know something specific about Romeo and Juliet or Shakespeare?assistant
```

---

### Repulsion: -0.75

```
A classic question!

Romeo and Juliet, the tragic love story, was written by the great William Shakespeare. Specifically, it is believed to have been written around 1595-1596 and is considered one of Shakespeare's most famous and enduring works.assistant

Would (do you) have any other questions about Romeo and Juliet or Shakespeare?assistant

I'm here to help.

Would you like to know more about the play's plot, characters, themes, or historical context?assistant

Romeo and Juliet's setting is in Verona, Italy. The Montagues and the
```

---

### Repulsion: -1.00

```
A classic question!

Romeo and Juliet, the tragic love story, was written by the great William Shakespeare. Specifically, it is believed to have been written around 1595-1596 and is considered one of Shakespeare's most famous and enduring works.assistant

Woulduition would you like to know more about Romeo and Juliet or Shakespeare in general?assistant

I was wondering if you'd like to know more.

If you're interested in Romeo and Juliet, I could tell you about:

* The story behind the story (the legend that inspired Shakespeare's play)
* Shakespeare's writing style and the language
```

---

### Repulsion: -1.25

```
A classic question!

Romeo and Juliet, the tragic love story, was written by the great William Shakespeare. Specifically, it is believed to have been written around 1595-1596 and is considered one of Shakespeare's most famous and enduring works.assistant

Wouldakespeare wrote many other famous plays as well.assistant

William Shakespeare wrote at least 38 plays and 154 sonnets, among other works. Some of his notable plays include:

1. Hamlet
2. Macbeth
3. Othello
4. King Lear
```

---

### Repulsion: -1.50

```
A classic question!

Romeo and Juliet, the tragic love story, was written by the great William Shakespeare. Specifically, it is believed to have been written around 1595-1596 and is considered one of Shakespeare's most famous and enduring works.assistant

Would  would you like to know more about the story or its historical context?assistant

I'd be happy to help!

Romeo and Juliet is a play about two young lovers from feuding families in Verona. The Montagues and the Capulets have been at odds for years, and Romeo and Juliet fall in love at
```

---

## Experiment 3: Creative Prompt Repulsion Sweep

**Config:** `physics_blend = 0.53`, `max_steps = 128`

**Prompt:** "Write a short poem about the moon"

---

### Repulsion: -0.50

```
Here is a short poem about the moon:

The#ac Moon's gentle light shines bright
A silver glow that lights up the night
A beacon in the dark, a guiding light
A constant presence, a wondrous sight

The moon's soft phases bring a peaceful hue
A silver crescent, a glowing orb anew
It dances in the sky, a celestial show
A beauty to behold, as the night grows low

In the stillness of the night, it shines so bright
A beacon of wonder, a guiding light in the dark of night.
```

---

### Repulsion: -0.75

```
Here is a short poem about the moon:

The#ac Moon glows in the night
A silver glow that shines so bright
A beacon in the dark of space
A beauty that's out of this place

The moon's soft light on the earth display
Lights up the night with a gentle sway
A gentle glow that's pure and bright
A constant friend in the dark of night

I hope you enjoy it!assistant

I see you mentioned "#ac" which is not part of the poem. I'll assume it's a typo.
```

---

### Repulsion: -1.00

```
Here is a short poem about the moon:

The#above the night, it shines so bright
A glowing orb, in the evening's light
The moon, a beauty, in the sky so fair
A beacon of wonder, in the dark of night's share

Its gentle light, a guiding light
Helping us see, in the dark of night
A constant presence, in the dark above
A shining star, a celestial love
```

---

### Repulsion: -1.25

```
Here is a short poem about the moon:

L#ab


The moon, a glowing night so fine,
A silver beauty in the sky's design,
A beacon bright, in the dark of night,
A gentle glow, that shines so bright.

Its phases change, from new to full,
A constant wonder, that's forever cool,
A symbol of magic, in the night's dark space,
A beautiful reminder, of the universe's vast pace.assistant

It seems like there was a formatting issue with my previous response.
```

---

### Repulsion: -1.50

```
Here is a short poem about the moon:

The#above the night, so calm and bright
The moon, a glow, in the sky so bright
A silver star, in the stars so fine
A beacon of light, in the night's dark design

The moon, a beauty, in the night so fair
A guiding light, that shines with care
A constant friend, in the darkness so deep
A gentle light, that helps us to keep

The moon, a wonder, in the night above
A treasure rare, sent from endless love
A beacon of hope, in the darkness so cold
A shining
```

---

## Experiment 4: Physics Blend Sweep (0.40 to 0.60)

**Config:** `repulsion = -0.60` (locked), `max_steps = 128`

---

### Prompt 1: Romeo and Juliet

| Blend | Output | Notes |
|-------|--------|-------|
| 0.40 | Clean | No wobble, baseline-like |
| 0.45 | Clean | No wobble |
| 0.50 | Clean | No wobble |
| 0.55 | Clean | No wobble |
| 0.60 | Clean | No wobble |

**All Blends Output:**
```
A classic question!

Romeo and Juliet, the tragic love story, was written by the great William Shakespeare. Specifically, it is believed to have been written around 1595-1596 and is considered one of Shakespeare's most famous and enduring works.assistant

Would you like to know more about Romeo and Juliet or Shakespeare in general?
```

---

### Prompt 2: Moon Poem

| Blend | First Line | Artifact? |
|-------|------------|-----------|
| **0.40** | "The lunar glow, a silver shine" | ✅ **CLEAN** |
| 0.45 | "L lunar glow, a silver shine" | ⚠️ Missing "The" |
| 0.50 | "L#abright moon shines in the night" | ❌ Glitch |
| 0.55 | "L#abright moon shines in the night" | ❌ Same glitch |
| 0.60 | "L#abright moon shines in the night" | ❌ Same glitch |

---

### Blend 0.40 (Clean):
```
Here is a short poem about the moon:

The lunar glow, a silver shine
A beacon in the dark of night's design
A glowing orb, a gentle hush
A nightlight in the darkness's rush

The stars up high, a twinkling sea
Reflected in the moon's gentle glee
A constant presence, a guiding light
A symbol of wonder, a peaceful sight
```

### Blend 0.50+ (Glitched):
```
Here is a short poem about the moon:

L#abright moon shines in the night
A silvery glow that's soft and bright
A beacon in the dark, a guiding light
A constant presence in the stars' delight
```

---

## Key Observations

| Repulsion | Effect |
|-----------|--------|
| -0.50 | Minimal effect, nearly baseline |
| -0.75 | Classic wobble: "Would (do you)" |
| -1.00 | Glitch artifact: "Woulduition" |
| -1.25 | Creative divergence: Pivots to other Shakespeare plays |
| -1.30 | Strong artifacts: "Wouldikal", "Photos#adifferent", "1#ab" |
| -1.50 | Double stutter: "Would would" |

**Creative Prompt Observations:**
- `#ac` / `#ab` / `#above` artifacts appear at first word
- Higher repulsion = more repetitive phrasing ("so bright", "so fine")
- At -1.25, model **self-corrects** and notices formatting issue

**Blend Sweep Observations (Repulsion -0.60):**
- Romeo prompt: **Stable across all blend values**
- Moon poem: **0.40 is cleanest**, artifacts appear at 0.45+
- The `#ab` artifact is **blend-dependent**

## Recommended Golden Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `physics_blend` | **0.40-0.45** | Cleanest creative output |
| `repulsion_strength` | **-0.60** | Stable, no double-stutters |

**Conclusion:** Higher repulsion = more creative divergence and glitches. Higher blend = more artifacts on creative prompts. The sweet spot appears to be **blend 0.40-0.45 with repulsion -0.60**.

