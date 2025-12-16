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


@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    """Generate text using the Niodoo physics engine."""
    start_time = time.time()
    
    cmd = [
        NIODOO_BINARY,
        "--model-path", MODEL_PATH,
        "--prompt", req.prompt,
        "--max-steps", str(req.max_tokens),
        "--seed", str(req.seed),
        "--physics-blend", str(req.physics_blend),
        f"--repulsion-strength={req.repulsion}"
    ]
    
    logger.info(f"Generating: {req.prompt[:60]}...")
    
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
        
        for line in result.stdout.split('\n'):
            m = token_pattern.search(line)
            if m:
                token_content = m.group(1).replace("\\n", "\n").replace("\\'", "'")
                output_text += token_content
                token_count += 1
        
        latency = time.time() - start_time
        logger.info(f"Generated {token_count} tokens in {latency:.2f}s")
        
        return GenerateResponse(
            text=output_text,
            latency_sec=round(latency, 3),
            tokens_generated=token_count,
            config={
                "physics_blend": req.physics_blend,
                "repulsion": req.repulsion,
                "seed": req.seed
            }
        )
        
    except subprocess.TimeoutExpired:
        logger.error("Generation timeout")
        raise HTTPException(status_code=504, detail="Generation timed out")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


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
    uvicorn.run(app, host="0.0.0.0", port=8000)
