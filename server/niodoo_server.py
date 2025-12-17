#!/usr/bin/env python3
"""
Niodoo v1.0 API Server
Gravitational Inference Engine for Large Language Models

This server provides a REST API for text generation using the Niodoo physics engine.
"""

import subprocess
import time
import json
import logging
import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# =============================================================================
# NIODOO v1.0 GOD ZONE PARAMETERS
# =============================================================================
NIODOO_PHYSICS_BLEND = 0.55
NIODOO_REPULSION = -0.60
DEFAULT_MAX_TOKENS = 128
DEFAULT_SEED = 123

# =============================================================================
# CONFIGURATION
# =============================================================================
NIODOO_BINARY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../target/release/niodoo")
)
MODEL_PATH = os.environ.get(
    "NIODOO_MODEL_PATH",
    "/home/ruffian/SplatRag/models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
)

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("niodoo")

# =============================================================================
# API
# =============================================================================
app = FastAPI(
    title="Niodoo v1.0",
    description="Gravitational Inference Engine for Large Language Models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


class GenerateRequest(BaseModel):
    """Request body for text generation."""
    prompt: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int = DEFAULT_SEED
    physics_blend: float = NIODOO_PHYSICS_BLEND
    repulsion: float = NIODOO_REPULSION


class GenerateResponse(BaseModel):
    """Response body for text generation."""
    text: str
    latency_sec: float
    tokens_generated: int
    config: dict
    telemetry: Optional[list] = None


@app.on_event("startup")
async def startup_event():
    """Verify binary exists on startup."""
    if not os.path.exists(NIODOO_BINARY):
        logger.error(f"Binary not found: {NIODOO_BINARY}")
        logger.error("Run ./scripts/INSTALL.sh to build the binary.")
    else:
        logger.info(f"Binary: {NIODOO_BINARY}")
        logger.info(f"Model: {MODEL_PATH}")
        logger.info(f"Config: blend={NIODOO_PHYSICS_BLEND}, repulsion={NIODOO_REPULSION}")
        logger.info("Niodoo v1.0 ready.")


# Telemetry Storage (In-Memory for Demo)
# Map: seed -> last_telemetry
last_telemetry_store = {}


class AutonomicRegulator:
    """
    Phase 4: The Heartbeat
    Self-regulates physics parameters based on the model's internal state (telemetry).
    """
    
    @staticmethod
    def regulate(seed: int, current_blend: float, current_repulsion: float) -> tuple[float, float, str]:
        """
        Analyze last turn's telemetry and return new (blend, repulsion, reason).
        """
        if seed not in last_telemetry_store:
            return current_blend, current_repulsion, "No history"
            
        trace = last_telemetry_store[seed]
        if not trace:
            return current_blend, current_repulsion, "Empty history"
            
        # 1. Calculate Metrics
        total_forces = [t.get("total_force", 0.0) for t in trace]
        if not total_forces:
            return current_blend, current_repulsion, "No force data"
            
        avg_force = sum(total_forces) / len(total_forces)
        glitch_count = sum(1 for t in trace if t.get("is_glitch", False))
        
        # Variance calculation
        variance = 0.0
        if len(total_forces) > 1:
            mean = avg_force
            variance = sum((x - mean) ** 2 for x in total_forces) / (len(total_forces) - 1)
            
        # 2. Apply Heuristics
        new_blend = current_blend
        new_repulsion = current_repulsion
        reason = []
        
        # HEURISTIC A: STRESS (Too much force/fighting)
        # If avg force > 15.0 or glitches detected -> Relax
        if avg_force > 15.0 or glitch_count > 2:
            new_blend = max(0.1, current_blend - 0.1)
            reason.append(f"High Stress (F={avg_force:.1f}, G={glitch_count}) -> Relaxed Blend")
            
        # HEURISTIC B: BOREDOM (Low variance/flat orbit)
        # If variance < 1.0 -> Increase Chaos (Repulsion)
        elif variance < 1.0:
            new_blend = min(1.0, current_blend + 0.05)
            # Increase repulsion (make it more negative)
            new_repulsion = max(-2.0, current_repulsion - 0.2) 
            reason.append(f"Boredom (Var={variance:.2f}) -> Boosted Chaos")
            
        if not reason:
            return current_blend, current_repulsion, "Homeostasis"
            
        return round(new_blend, 3), round(new_repulsion, 2), " + ".join(reason)

@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    """Generate text using the Niodoo physics engine."""
    
    # === PHASE 4: THE HEARTBEAT (Autonomic Regulation) ===
    # Calculate physics parameters based on previous state
    # Start with requested values (defaults if not provided)
    base_blend = req.physics_blend
    base_repulsion = req.repulsion
    
    # Ask the Regulator
    final_blend, final_repulsion, regulation_reason = AutonomicRegulator.regulate(
        req.seed, base_blend, base_repulsion
    )
    
    if regulation_reason != "Homeostasis" and regulation_reason != "No history":
        logger.info(f"❤️ HEARTBEAT ACTION: {regulation_reason}. Params: ({base_blend}, {base_repulsion}) -> ({final_blend}, {final_repulsion})")

    # === PHASE 3: THE MIRROR (Context Injection) ===
    # Check if this is a meta-cognitive question ("Why...?")
    meta_context = ""
    is_meta_query = "why" in req.prompt.lower()
    
    if is_meta_query and req.seed in last_telemetry_store:
        last_trace = last_telemetry_store[req.seed]
        
        # Summarize the physics of the last run
        # Find the top 3 most repelled/forced tokens
        forces = []
        for frame in last_trace:
            if "total_force" in frame and "token" in frame:
                forces.append((frame["total_force"], frame["token"], frame.get("repulsion_force", 0.0)))
        
        # Sort by total force descending
        forces.sort(key=lambda x: x[0], reverse=True)
        top_forces = forces[:3]
        
        # Construct the "Proprioception" injection
        meta_context = "\n[SYSTEM TELEMETRY INJECTION]:\n"
        meta_context += "In your last thought process, you felt the following internal physics:\n"
        for total, token, rep in top_forces:
            meta_context += f"- On word '{token.strip()}', you felt Total Force {total:.1f} (Repulsion: {rep:.1f}).\n"
        meta_context += "Use this data to explain your creative choices.\n\n"
        
        logger.info(f"Injecting Meta-Context: {meta_context.strip()}")

    # Prepend meta-context to prompt if it exists
    full_prompt = meta_context + req.prompt
    
    start_time = time.time()
    
    cmd = [
        NIODOO_BINARY,
        "--model-path", MODEL_PATH,
        "--prompt", full_prompt,
        "--max-steps", str(req.max_tokens),
        "--seed", str(req.seed),
        "--physics-blend", str(final_blend),
        f"--repulsion-strength={final_repulsion}",
        "--mode-orbital",
        "--orbit-speed", "0.15"
    ]
    
    logger.info(f"Generating: {full_prompt[:100]}...")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            logger.error(f"Engine error: {result.stderr[-500:]}")
            raise HTTPException(
                status_code=500,
                detail=f"Engine error: {result.stderr[-200:]}"
            )
        
        # Parse decoded tokens from output
        output_text = ""
        token_count = 0
        token_pattern = re.compile(r"\[DBG: Decoded '(.*)'\]")
        
        # Telemetry parsing (Phase 3: The Observer)
        telemetry_log = []
        
        # DEBUG: Log raw output glimpse
        logger.info(f"Raw Stdout Glimpse (First 500 chars): {result.stdout[:500]}")
        
        for line in result.stdout.split('\n'):
            # 1. Capture Text
            m = token_pattern.search(line)
            if m:
                token_content = m.group(1).replace("\\n", "\n").replace("\\'", "'")
                output_text += token_content
                token_count += 1
            
            # 2. Capture Physics (JSON lines)
            clean = line.strip()
            if clean.startswith("[TELEMETRY]"):
                try:
                    json_str = clean.replace("[TELEMETRY]", "").strip()
                    # Remove trailing comma if present (from the Rust array format)
                    clean_line = json_str.rstrip(',')
                    data = json.loads(clean_line)
                    telemetry_log.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON line: {clean[:40]}... Error: {e}")
                    pass
        
        # Store telemetry for next turn (The Mirror)
        if len(telemetry_log) > 0:
            last_telemetry_store[req.seed] = telemetry_log
        
        latency = time.time() - start_time
        logger.info(f"Generated {token_count} tokens in {latency:.2f}s | Physics Frames: {len(telemetry_log)}")
        
        return GenerateResponse(
            text=output_text,
            latency_sec=round(latency, 3),
            tokens_generated=token_count,
            config={
                "physics_blend": final_blend,
                "repulsion": final_repulsion,
                "regulation_reason": regulation_reason,
                "seed": req.seed
            },
            telemetry=telemetry_log
        )
        
    except subprocess.TimeoutExpired:
        logger.error("Generation timeout")
        raise HTTPException(status_code=504, detail="Generation timed out")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class DebugInjectRequest(BaseModel):
    seed: int
    telemetry: list


@app.post("/debug/inject")
def debug_inject(req: DebugInjectRequest):
    """Inject fake telemetry for testing autonomic regulation."""
    last_telemetry_store[req.seed] = req.telemetry
    logger.info(f"Injected fake telemetry for seed {req.seed} ({len(req.telemetry)} frames)")
    return {"status": "ok"}



@app.get("/health")
def health_check():
    """Health check endpoint."""
    binary_ok = os.path.exists(NIODOO_BINARY)
    model_ok = os.path.exists(MODEL_PATH)
    
    return {
        "status": "online" if binary_ok and model_ok else "degraded",
        "version": "1.0.0",
        "binary": binary_ok,
        "model": model_ok,
        "config": {
            "physics_blend": NIODOO_PHYSICS_BLEND,
            "repulsion": NIODOO_REPULSION
        }
    }


@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "name": "Niodoo v1.0",
        "description": "Gravitational Inference Engine",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
