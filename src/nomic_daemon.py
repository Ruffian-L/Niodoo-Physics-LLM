import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import os

# Configuration
MODEL_ID = os.environ.get("SPLATRAG_MODEL", "nomic-ai/nomic-embed-text-v1.5")
# We default to 768 (no truncation) so Rust can handle it.
# If user wants to force it here, they can set the env var.
MATRYOSHKA_DIM = int(os.environ.get("SPLATRAG_EMBED_DIM", "768")) 

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    # Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🐍 Python Executable: {sys.executable}", file=sys.stderr)
    try:
        import sentencepiece
        print("✅ SentencePiece imported successfully", file=sys.stderr)
    except ImportError:
        print("❌ SentencePiece import failed", file=sys.stderr)

    embedder_type = os.environ.get("SPLATRAG_EMBEDDER", "nomic")
    print(f"🔌 Initializing Embedder: {embedder_type}", file=sys.stderr)

    # Flag to indicate if SentenceTransformer's encode method should be used
    use_sentence_transformer = False

    try:
        if embedder_type == "siglip":
            # SigLIP Text Tower
            # SigLIP Text Tower
            model_id = "google/siglip-base-patch16-224"
            # Try fast tokenizer first
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            except Exception as e:
                print(f"⚠️ Fast tokenizer failed: {e}, trying slow...", file=sys.stderr)
                tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
            
            # Load only the text model
            model = AutoModel.from_pretrained(model_id).text_model
            model.to(device)
            if device == "cuda":
                model.half()
            model.eval()
        elif embedder_type == "mxbai":
            # ==== MXBAI - BEST WORD-LEVEL TEXT EMBEDDER DEC 2025 ====
            # Use SentenceTransformer for clean pooled 1024-dim output (no manual pooling needed)
            from sentence_transformers import SentenceTransformer
            model_id = "mixedbread-ai/mxbai-embed-large-v1"
            print(f"🔌 Initializing Embedder: mxbai (SentenceTransformer, 1024-dim)", file=sys.stderr)
            model = SentenceTransformer(model_id, truncate_dim=1024)
            model = model.to(device)
            tokenizer = None  # Not needed for SentenceTransformer
            use_sentence_transformer = True
            if device == "cuda":
                model.half()
                try:
                    import flash_attn
                    print("Flash Attention 2 detected - embeddings will be 2-3x faster", file=sys.stderr)
                except ImportError:
                    print("No Flash Attention - install with: pip install flash-attn --no-build-isolation", file=sys.stderr)
            model.eval()

        elif embedder_type == "e5-mistral":
            # E5-Mistral
            # Requires specific prompting and EOS pooling
            # Requires specific prompting and EOS pooling
            model_id = "intfloat/e5-mistral-7b-instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
            model.to(device)
            if device == "cuda":
                model.half()
            model.eval()
        else:
            # Default Nomic
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
            model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
            model.to(device)
            if device == "cuda":
                model.half()
            model.eval()
    except Exception as e:
        print(f"Error loading model {embedder_type}: {e}", file=sys.stderr)
        sys.exit(1)

    # Loop
    for line in sys.stdin:
        if not line:
            break
        
        try:
            req = json.loads(line)
            texts = req.get("texts", [])
            mode = req.get("mode", "search_query")
            
            if not texts:
                print("[]", flush=True)
                continue

            # Prefix handling (task-specific prompts)
            prefixed_texts = []
            for t in texts:
                if mode == "search_document":
                    prefixed_texts.append("search_document: " + t)
                elif mode == "search_query":
                    if embedder_type == "mxbai":
                        prompt = "Represent this sentence for searching relevant passages: "
                        prefixed_texts.append(prompt + t)
                    elif embedder_type == "e5-mistral":
                        task = "Given a web search query, retrieve relevant passages that answer the query"
                        prefixed_texts.append(f"Instruction: {task}\\nQuery: {t}")
                    else:
                        prefixed_texts.append("search_query: " + t)
                else:
                    prefixed_texts.append(t)
            
            # ===== SENTENCE TRANSFORMER PATH (mxbai clean 1024-dim) =====
            if use_sentence_transformer:
                # SentenceTransformer handles tokenization + forward + pooling + normalize
                pooled_embeddings = model.encode(
                    prefixed_texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=False,  # Return torch tensor
                    convert_to_tensor=True,   # CRITICAL: return tensor not list
                    batch_size=len(prefixed_texts)  # Process all at once
                )
                # pooled_embeddings: torch.Tensor [batch, 1024]
                print(f"DEBUG: SentenceTransformer output shape: {pooled_embeddings.shape}", file=sys.stderr)
                
                # Format response
                resp = []
                for i in range(len(texts)):
                    resp.append({
                        "pooled": pooled_embeddings[i].cpu().tolist(),
                        "tokens": [],  # Not available in ST mode
                        "token_embeddings": []
                    })
                
                # Send response and continue to next batch
                output = json.dumps(resp)
                print(output, flush=True)
                continue
            
            # ===== MANUAL FORWARD PASS (for other embedders) =====

            # Tokenize
            # E5-Mistral needs left padding for generation but right padding for embedding usually?
            # Actually, for batch embedding with last-token pooling, we need to be careful with padding.
            # If we right-pad, the last token is [PAD]. We need to find the last REAL token.
            # Or use left-padding?
            # Transformers tokenizer handles padding.
            
            if embedder_type == "e5-mistral":
                tokenizer.padding_side = "right" # Standard
                # Ensure EOS token is added?
                # Mistral tokenizer usually adds EOS?
                # Let's check.
                pass

            inputs = tokenizer(prefixed_texts, padding=True, truncation=True, max_length=512 if embedder_type == "mxbai" else (4096 if embedder_type == "e5-mistral" else 2048), return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
                if embedder_type == "siglip":
                    # SigLIP has a specific pooler_output
                    # outputs.pooler_output is [batch, hidden]
                    # But wait, SigLIP text model output might be different?
                    # The user snippet says: embeddings = outputs.pooler_output
                    # Let's verify if pooler_output exists.
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        pooled = outputs.pooler_output
                    else:
                        # Fallback to last_hidden_state + pooler if needed, or just use last token?
                        # SigLIP usually uses EOS token pooling.
                        # Let's assume pooler_output is correct as per user snippet.
                        pooled = outputs.last_hidden_state[:, 0] # CLS/EOS? SigLIP uses EOS usually.
                        # Actually, let's stick to user snippet: outputs.pooler_output
                        # If it fails, we catch exception.
                        pooled = outputs.pooler_output
                    
                    # SigLIP embeddings are usually normalized?
                    # Let's normalize anyway.
                    pooled = F.normalize(pooled, p=2, dim=1)
                    
                    # Token embeddings for "Splat" visualization?
                    # SigLIP is not really designed for token-level embeddings in the same way,
                    # but we can use last_hidden_state.
                    token_embeddings = outputs.last_hidden_state

                elif embedder_type == "e5-mistral":
                    # E5-Mistral: Last Token Pooling
                    # We need to find the index of the last non-padding token.
                    # attention_mask is 1 for tokens, 0 for padding.
                    # sum(mask) - 1 is the index of the last token.
                    
                    token_embeddings = outputs.last_hidden_state # [batch, seq, hidden]
                    # input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float() # Not needed for last token pooling
                    
                    # Last token index
                    # sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
                    # batch_size = token_embeddings.shape[0]
                    # pooled = token_embeddings[torch.arange(batch_size, device=token_embeddings.device), sequence_lengths]
                    
                    # Robust last token pooling
                    # Gather the last token in each sequence
                    # (batch_size, hidden_size)
                    
                    # Note: Mistral might append EOS?
                    # If tokenizer adds EOS, it's the last token.
                    
                    sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1
                    pooled = token_embeddings[torch.arange(token_embeddings.shape[0], device=device), sequence_lengths]

                    pooled = F.normalize(pooled, p=2, dim=1)
                
                elif embedder_type == "mxbai":
                    # mxbai uses CLS token (index 0) + their custom projection head
                    # pooler_output SHOULD be 1024-D
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        pooled = outputs.pooler_output
                        print(f"DEBUG: mxbai pooler_output shape: {pooled.shape}", file=sys.stderr)
                    else:
                        # Fallback to CLS token from last_hidden_state
                        pooled = outputs.last_hidden_state[:, 0]
                        print(f"DEBUG: mxbai using CLS token, shape: {pooled.shape}", file=sys.stderr)
                    
                    # Check dimension and pad if needed
                    if pooled.shape[1] < 1024:
                        print(f"WARNING: mxbai output is {pooled.shape[1]}-D, padding to 1024", file=sys.stderr)
                        padding = torch.zeros(pooled.shape[0], 1024 - pooled.shape[1], device=pooled.device)
                        pooled = torch.cat([pooled, padding], dim=1)
                    elif pooled.shape[1] > 1024:
                        print(f"WARNING: mxbai output is {pooled.shape[1]}-D, truncating to 1024", file=sys.stderr)
                        pooled = pooled[:, :1024]
                    
                    # Normalize
                    pooled = F.normalize(pooled, p=2, dim=1)
                    
                    # Token-level embeddings = last_hidden_state (pre-projection)
                    token_embeddings = outputs.last_hidden_state
                    
                else:
                    # Nomic / BERT-like
                    # last_hidden_state: [batch, seq, 768]
                    token_embeddings = outputs.last_hidden_state
                    
                    # Nomic uses mean pooling.
                    pooled = mean_pooling(token_embeddings, inputs['attention_mask'])

                    pooled = F.layer_norm(pooled, normalized_shape=(pooled.shape[1],))
                    pooled = F.normalize(pooled, p=2, dim=1)
                
                # Matryoshka slicing (Optional in Python, enforced in Rust)
                if MATRYOSHKA_DIM < pooled.shape[1]:
                    pooled = pooled[:, :MATRYOSHKA_DIM]
                    pooled = F.normalize(pooled, p=2, dim=1)
                    # We do NOT truncate token_embeddings here, as Rust might need 768D for RVQ residuals.

            # Output
            resp = []
            for i in range(len(texts)):
                # Get valid tokens only (remove padding)
                # attention_mask[i] is [seq_len]
                mask = inputs['attention_mask'][i].bool()
                valid_tokens_emb = token_embeddings[i][mask] # [real_seq, 768]
                
                # Convert tokens to strings for debugging/alignment
                # input_ids[i][mask]
                token_ids = inputs['input_ids'][i][mask]
                tokens_str = tokenizer.convert_ids_to_tokens(token_ids)

                resp.append({
                    "pooled": pooled[i].float().cpu().tolist(),
                    "token_embeddings": valid_tokens_emb.float().cpu().tolist(), 
                    "tokens": tokens_str
                })
            
            print(json.dumps(resp), flush=True)

        except Exception as e:
            print(f"Error processing batch: {e}", file=sys.stderr)
            print("[]", flush=True)

if __name__ == "__main__":
    main()
