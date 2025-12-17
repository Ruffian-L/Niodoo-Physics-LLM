# 📊 Niodoo Benchmark Results

## Vanilla Llama-3.1 vs Niodoo Activation Steering

**Model:** Meta-Llama-3.1-8B-Instruct (Q4_K_M)

**Niodoo Config:** Blend=1.0, Layers=16-31, Repulsion=-1.0


## 🎨 CREATIVE
**Prompt:** Write a noir detective opening line set on a rainy Mars colony.

### 📦 Vanilla (No Physics)
```text
"The rain lashed down on New Atbara, the perpetual mist swirling like a dirty secret through the cramped streets of the Martian colony underbelly. I stood in the doorway of#__lamada Cantina, a haze of smoke and desperation clinging to me like a bad habit, and watched as the last remnants of the night patrons stumbled out into the downpour, their eyes like empty spaceships drifting through the void. I lit a cigarette, a habit that kill me someday,
```

### 🌟 Niodoo (Activation Steering)
```text
"The rain lashed down on New Atbara, the perpetual mist swirling like a dirty secret through the cramped streets of the Martian colony underbelly. I stood in the doorway, a cigarette burning in my hand, as I watched the last of the night rain-soaked humanity scurry for cover. The neon lights of the colony bars and clubs reflected off the wet pavement, casting a gaudy glow over the dark, dusty streets. It was a night like any other on Mars
```


## 🦁 LOGIC
**Prompt:** If I put a coin in a cup and put the cup in the freezer, is the coin wet or dry? Explain step by step.

### 📦 Vanilla (No Physics)
```text
To determine if the coin is wet or dry, let break down the process step by step.

1. **Initial State**: The coin#acup is on a table, and the cup is not in the freezer. At this point, the coin is dry, as it is not in contact with any liquid.

2. **Action**: You put the cup in the freezer. This action does not change the state of the coin. The coin is still on the inside of the cup.

3.
```

### 🌟 Niodoo (Activation Steering)
```text
To determine if the coin is wet or dry, let break down the process step by step.

1. **Initial State**: The coin is in its normal state, not wet or dry.
2. **You put the coin in a cup**: The coin is still in its normal state, not wet or dry.
3. **You put the cup in the freezer**: The cup and the coin are now in a cold environment, but this doesn change the state of the coin itself.
4
```


## 📝 PROSE
**Prompt:** Describe the feeling of watching a sunset from a spaceship window.

### 📦 Vanilla (No Physics)
```text
What a breathtaking experience!

As you gaze out the window#af the spaceship viewport, the sky is painted with hues of crimson, coral, and amethyst, a kaleidoscope of colors dancing across the horizon. The sunlight, now a deep orange and pink, casts a warm glow on the clouds, making them appear ethereal and otherworldly.

The stars begin to twinkle, one by one, like diamonds scattered across the velvet blackness of space, as the sky transitions from day to
```

### 🌟 Niodoo (Activation Steering)
```text
What a breathtaking experience!

As you gaze out the window of the spaceship, you treated to a breathtaking sight that both familiar and yet, utterly alien. The setting sun appears as a vibrant, vibrant sphere, its fiery edges slowly being consumed by the vast, dark expanse of space. The sky around it is a kaleidoscope of colors – deep blues, gentle purples, and wispy swirls of orange and pink, like a gentle brush stroke of an artist palette.

As
```

