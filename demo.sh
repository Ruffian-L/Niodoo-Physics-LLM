#!/bin/bash
# ============================================================================
# NIODOO v3.1 DEMO
# Demonstrates self-correction on reasoning problems
# Usage: ./demo.sh
#
# NOTE: LLMs are non-deterministic. Vanilla may sometimes get answers right
# on a given run. Run multiple times to see variance. Niodoo shows visible
# "wobble" reasoning even when reaching the same conclusion.
# ============================================================================

set -e

MODEL_PATH="${NIODOO_MODEL:-/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}"
BINARY="./target/release/niodoo"

# Function to check if answer contains correct response
check_towels() {
    local output="$1"
    if echo "$output" | grep -qiE "(1 hour|one hour|takes 1|\\$1\\$|boxed{1})" && \
       ! echo "$output" | grep -qiE "(50 hours|50 x 1|fifty hours)"; then
        echo "CORRECT (1 hour)"
    elif echo "$output" | grep -qiE "(50 hours|50 x 1|fifty hours)"; then
        echo "WRONG (50 hours)"
    else
        echo "UNCLEAR"
    fi
}

check_monty() {
    local output="$1"
    if echo "$output" | grep -qiE "(2/3|2 / 3|66|67|two.thirds)"; then
        echo "CORRECT (2/3)"
    elif echo "$output" | grep -qiE "(50.50|50-50|50%|1/2|equal|doesn.t matter)"; then
        echo "WRONG (50-50)"
    else
        echo "UNCLEAR"
    fi
}

echo ""
echo "================================================================"
echo "  NIODOO v3.1 DEMO"
echo "================================================================"
echo ""
echo "NOTE: LLMs are non-deterministic. Results may vary between runs."
echo "      Run multiple times to observe variance."
echo ""

# ============================================================================
# TEST 1: DRYING TOWELS
# ============================================================================
PROMPT1="It takes 1 hour to dry one towel on a sunny clothesline. How long does it take to dry 50 towels?"

echo "TEST 1: DRYING TOWELS"
echo "---------------------"
echo "Prompt: \"$PROMPT1\""
echo "Correct: 1 hour (parallel drying)"
echo ""

echo "[Vanilla Llama 3.1]"
if command -v ollama &> /dev/null; then
    VANILLA1=$(ollama run llama3.1 "$PROMPT1" 2>/dev/null | head -30)
    echo "$VANILLA1"
    echo ""
    VERDICT1=$(check_towels "$VANILLA1")
    echo "Result: $VERDICT1"
else
    echo "[Ollama not installed]"
    VERDICT1="SKIPPED"
fi
echo ""

echo "[Niodoo v3.1]"
if [ -f "$BINARY" ]; then
    NIODOO1=$($BINARY --model-path "$MODEL_PATH" --prompt "$PROMPT1" --mode-orbital \
        --physics-blend 1.5 --repulsion-strength=-0.5 --gravity-well 0.2 \
        --orbit-speed 0.1 --max-steps 512 --seed 42 2>/dev/null | \
        grep "DBG: Decoded" | sed "s/\[DBG: Decoded '//g" | sed "s/'\]//g" | \
        sed 's/\\n/\n/g' | tr -d '\n' | sed 's/  */ /g' | fold -s -w 80)
    echo "$NIODOO1"
    echo ""
    NIODOO_VERDICT1=$(check_towels "$NIODOO1")
    echo "Result: $NIODOO_VERDICT1"
else
    echo "[Binary not found - build with: cargo build --release --bin niodoo]"
    NIODOO_VERDICT1="SKIPPED"
fi
echo ""
echo "----------------------------------------------------------------"

# ============================================================================
# TEST 2: MONTY HALL
# ============================================================================
PROMPT2="You're on a game show. There are 3 doors. Behind one is a car, behind the others are goats. You pick door 1. The host opens door 3 to reveal a goat. Should you switch to door 2 or stick with door 1? What gives you better odds?"

echo ""
echo "TEST 2: MONTY HALL"
echo "------------------"
echo "Prompt: \"$PROMPT2\""
echo "Correct: Switch - gives 2/3 (66.7%) chance"
echo ""

echo "[Vanilla Llama 3.1]"
if command -v ollama &> /dev/null; then
    VANILLA2=$(ollama run llama3.1 "$PROMPT2" 2>/dev/null | head -30)
    echo "$VANILLA2"
    echo ""
    VERDICT2=$(check_monty "$VANILLA2")
    echo "Result: $VERDICT2"
else
    echo "[Ollama not installed]"
    VERDICT2="SKIPPED"
fi
echo ""

echo "[Niodoo v3.1]"
if [ -f "$BINARY" ]; then
    NIODOO2=$($BINARY --model-path "$MODEL_PATH" --prompt "$PROMPT2" --mode-orbital \
        --physics-blend 1.5 --repulsion-strength=-0.5 --gravity-well 0.2 \
        --orbit-speed 0.1 --max-steps 512 --seed 42 2>/dev/null | \
        grep "DBG: Decoded" | sed "s/\[DBG: Decoded '//g" | sed "s/'\]//g" | \
        sed 's/\\n/\n/g' | tr -d '\n' | sed 's/  */ /g' | fold -s -w 80)
    echo "$NIODOO2"
    echo ""
    NIODOO_VERDICT2=$(check_monty "$NIODOO2")
    echo "Result: $NIODOO_VERDICT2"
else
    echo "[Binary not found]"
    NIODOO_VERDICT2="SKIPPED"
fi
echo ""
echo "----------------------------------------------------------------"

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================"
echo "SUMMARY (this run)"
echo "================================================================"
echo ""
echo "  Problem        | Vanilla          | Niodoo"
echo "  -------------  | ---------------- | ----------------"
echo "  Drying Towels  | $VERDICT1 | $NIODOO_VERDICT1"
echo "  Monty Hall     | $VERDICT2 | $NIODOO_VERDICT2"
echo ""
echo "NOTE: Results vary between runs. Vanilla sometimes gets it right."
echo "      The key difference is Niodoo shows visible reasoning (wobble)."
echo ""
echo "================================================================"
echo ""
