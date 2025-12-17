#!/bin/bash
# =============================================================================
# NIODOO v2.0 INSTALLER
# Inference-Time Activation Steering for Large Language Models
# =============================================================================
set -e

echo "=============================================="
echo "  NIODOO v2.0 INSTALLER"
echo "  Activation Steering Engine"
echo "=============================================="
echo ""

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Check Rust
echo "[1/4] Checking Rust installation..."
if ! command -v cargo &> /dev/null; then
    echo "  -> Cargo not found. Installing Rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "  -> Rust installed successfully."
else
    RUST_VERSION=$(rustc --version)
    echo "  -> Found: $RUST_VERSION"
fi

# Build Rust binary
echo ""
echo "[2/4] Building Core Engine..."
if [ ! -f "Cargo.toml" ]; then
    echo "ERROR: Cargo.toml not found. Run this script from the Niodoo_Release directory."
    exit 1
fi

cargo build --release --bin niodoo
BINARY_PATH="$PROJECT_ROOT/target/release/niodoo"

if [ -f "$BINARY_PATH" ]; then
    echo "  -> Binary: $BINARY_PATH"
else
    echo "ERROR: Build failed. Binary not found."
    exit 1
fi

# Setup Python environment
echo ""
echo "[3/4] Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  -> Virtual environment created."
else
    echo "  -> Virtual environment exists."
fi

source venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "  -> Dependencies installed."

# Verify installation
echo ""
echo "[4/4] Verifying installation..."
if [ -f "$BINARY_PATH" ] && source venv/bin/activate && python3 -c "import fastapi" 2>/dev/null; then
    echo "  -> All components verified."
else
    echo "WARNING: Some components may not be fully installed."
fi

# Print summary
echo ""
echo "=============================================="
echo "  INSTALLATION COMPLETE"
echo "=============================================="
echo ""
echo "Binary:  $BINARY_PATH"
echo "Python:  $PROJECT_ROOT/venv"
echo ""
echo "v2.0 Validated Configuration:"
echo "  physics_blend:        1.0"
echo "  repulsion_strength:  -1.0"
echo "  physics_start_layer: 16 (skip syntax)"
echo "  physics_end_layer:   31 (semantic only)"
echo "  orbit_speed:         0.15"
echo "  gravity_well:        1.0"
echo ""
echo "Usage:"
echo ""
echo "  Start API Server:"
echo "    source venv/bin/activate"
echo "    python3 server/niodoo_server.py"
echo ""
echo "  Run Benchmark:"
echo "    source venv/bin/activate"
echo "    python3 benchmarks/run_benchmark.py benchmarks/prompts.json"
echo ""
echo "  Command Line:"
echo "    ./target/release/niodoo \\"
echo "      --model-path /path/to/model.gguf \\"
echo "      --prompt \"Your prompt here\" \\"
echo "      --mode-orbital \\"
echo "      --physics-blend 1.0 \\"
echo "      --repulsion-strength -1.0"
echo ""
echo "=============================================="

