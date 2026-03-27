"""
MoE Expert Sniper v3 — HF Transformers backbone + selective weight injection.

The OOM fix: never allocate expert weights on GPU. Use accelerate's
init_empty_weights to create a zero-memory skeleton, then inject only
non-expert weights. Monkey-patch MoE forward for expert sniping.

Usage:
    python3 sniper_122b_v3.py --model-dir /workspace/qwen35-122b-stream \
        --original-dir /workspace/qwen35-122b-a10b-4bit
"""

import os
import gc
import json
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

BITS = 4
GROUP_SIZE = 64


def dequantize_4bit(weight, scales, biases, group_size=64):
    """Dequantize MLX 4-bit to bfloat16."""
    if weight.dtype not in (torch.uint32, torch.int32):
        return weight.to(torch.bfloat16)
    out_features = weight.shape[0]
    w = weight.to(torch.int32)
    unpacked = torch.stack([(w >> (4 * i)) & 0xF for i in range(8)], dim=-1)
    in_features = unpacked.shape[1] * 8
    unpacked = unpacked.reshape(out_features, in_features).float()
    num_groups = in_features // group_size
    unpacked = unpacked.reshape(out_features, num_groups, group_size)
    dequantized = unpacked * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
    return dequantized.reshape(out_features, in_features).to(torch.bfloat16)


class ExpertSniper:
    """Loads active MoE experts from NVMe or VRAM cache."""

    def __init__(self, expert_dir, num_layers, device="cuda", cache_layers=15):
        self.expert_dir = Path(expert_dir)
        self.device = device
        self.handles = {}
        self.vram_cache = {}
        self.cache_layers = cache_layers
        self.num_layers = num_layers

    def cache_in_vram(self):
        print(f"  Caching expert layers 0-{self.cache_layers-1} in VRAM...")
        t0 = time.time()
        for i in range(min(self.cache_layers, self.num_layers)):
            path = self.expert_dir / f"layer_{i:02d}.safetensors"
            if not path.exists():
                continue
            data = {}
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    data[k] = f.get_tensor(k).to(self.device)
            self.vram_cache[i] = data
        gb = sum(sum(t.nbytes for t in d.values()) for d in self.vram_cache.values()) / 1e9
        print(f"  Cached: {gb:.2f} GB [{time.time()-t0:.1f}s]")

    def _handle(self, layer_idx):
        if layer_idx not in self.handles:
            self.handles[layer_idx] = safe_open(
                str(self.expert_dir / f"layer_{layer_idx:02d}.safetensors"),
                framework="pt", device="cpu"
            )
        return self.handles[layer_idx]

    def get_experts(self, layer_idx, expert_ids):
        """Get dequantized [top_k, out, in] weight tensors for active experts."""
        ids = expert_ids if isinstance(expert_ids, list) else expert_ids.tolist()
        result = {}

        if layer_idx in self.vram_cache:
            data = self.vram_cache[layer_idx]
            idx = torch.tensor(ids, dtype=torch.long, device=self.device)
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                w = torch.index_select(data[f"{proj}.weight"], 0, idx)
                s = torch.index_select(data[f"{proj}.scales"], 0, idx)
                b = torch.index_select(data[f"{proj}.biases"], 0, idx)
                result[proj] = dequantize_4bit(w, s, b, GROUP_SIZE)
        else:
            h = self._handle(layer_idx)
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                fw = h.get_tensor(f"{proj}.weight")
                fs = h.get_tensor(f"{proj}.scales")
                fb = h.get_tensor(f"{proj}.biases")
                w = torch.stack([fw[i] for i in ids]).to(self.device)
                s = torch.stack([fs[i] for i in ids]).to(self.device)
                b = torch.stack([fb[i] for i in ids]).to(self.device)
                result[proj] = dequantize_4bit(w, s, b, GROUP_SIZE)
        return result


def patch_moe_layer(moe_block, layer_idx, sniper, top_k):
    """Replace MoE forward to snipe experts from NVMe instead of VRAM."""
    original_gate = moe_block.gate
    shared_expert = getattr(moe_block, 'shared_expert', None)
    shared_expert_gate = getattr(moe_block, 'shared_expert_gate', None)

    def sniped_forward(hidden_states):
        B, L, D = hidden_states.shape
        x = hidden_states.view(-1, D)

        # Route
        router_logits = original_gate(x)
        scores = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_w, topk_idx = torch.topk(scores, top_k, dim=-1)
        topk_w = (topk_w / topk_w.sum(dim=-1, keepdim=True)).to(hidden_states.dtype)

        # Unique experts needed
        needed = topk_idx.unique().tolist()
        expert_w = sniper.get_experts(layer_idx, needed)
        id_to_local = {eid: i for i, eid in enumerate(needed)}

        output = torch.zeros_like(x)
        for local_idx, eid in enumerate(needed):
            mask = (topk_idx == eid)
            token_mask = mask.any(dim=-1)
            tidx = token_mask.nonzero(as_tuple=True)[0]
            if len(tidx) == 0:
                continue
            w = (topk_w * mask.to(topk_w.dtype)).sum(dim=-1)

            inp = x[tidx]
            g = F.silu(inp @ expert_w["gate_proj"][local_idx].t())
            u = inp @ expert_w["up_proj"][local_idx].t()
            out = (g * u) @ expert_w["down_proj"][local_idx].t()
            output[tidx] += w[tidx].unsqueeze(-1) * out

        if shared_expert is not None:
            s_out = shared_expert(x)
            if shared_expert_gate is not None:
                s_out = s_out * torch.sigmoid(shared_expert_gate(x))
            output = output + s_out

        del expert_w
        return output.view(B, L, D)

    # Replace forward
    moe_block.forward = sniped_forward
    # Delete expert modules to free memory
    if hasattr(moe_block, 'experts') and moe_block.experts is not None:
        del moe_block.experts
        moe_block.experts = None
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="/workspace/qwen35-122b-stream")
    parser.add_argument("--original-dir", default="/workspace/qwen35-122b-a10b-4bit")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--cache-layers", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("  MoE EXPERT SNIPER v3")
    print("  HF Transformers backbone + Expert Sniping")
    print("=" * 60)

    device = args.device
    model_dir = Path(args.model_dir)
    original_dir = Path(args.original_dir)

    # ── Step 1: Load config ──
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(str(original_dir), trust_remote_code=True)
    text_cfg = config.text_config if hasattr(config, 'text_config') else config
    num_layers = text_cfg.num_hidden_layers
    top_k = getattr(text_cfg, 'num_experts_per_tok', 8)
    print(f"  {num_layers} layers, top-{top_k} experts")

    # ── Step 2: Create model skeleton on CPU with NO expert weights ──
    print("\n[1/5] Creating model skeleton (no experts)...")
    t0 = time.time()

    # Keep num_experts=256 (router needs correct output dim) but shrink
    # expert intermediate to 1 so each expert is tiny (~12 KB instead of 18 MB)
    orig_intermediate = text_cfg.moe_intermediate_size
    text_cfg.moe_intermediate_size = 1

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_config(
        text_cfg, trust_remote_code=True, dtype=torch.bfloat16
    )
    text_cfg.moe_intermediate_size = orig_intermediate  # restore

    print(f"  Created in {time.time()-t0:.1f}s")

    # ── Step 3: Inject dequantized pinned weights ──
    print("\n[2/5] Injecting pinned weights...")
    t0 = time.time()
    model = model.to(device)
    vram0 = torch.cuda.memory_allocated() / 1e9
    print(f"  Empty model VRAM: {vram0:.2f} GB")

    pinned_path = model_dir / "pinned.safetensors"
    model_params = dict(model.named_parameters())

    loaded = 0
    skipped = 0

    with safe_open(str(pinned_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        # Group quantized weights by base name
        bases = {}
        for k in keys:
            if k.endswith(".scales"):
                base = k[:-7]
                bases.setdefault(base, {})["scales"] = k
            elif k.endswith(".biases"):
                base = k[:-7]
                bases.setdefault(base, {})["biases"] = k
            elif k.endswith(".weight"):
                base = k[:-7]
                bases.setdefault(base, {})["weight"] = k
            else:
                bases.setdefault(k, {})["raw"] = k

        for base, parts in bases.items():
            # Try to find matching model param
            # The pinned keys have "language_model." prefix, model might not
            target_key = base + ".weight"
            alt_key = target_key.replace("language_model.", "", 1)

            param = model_params.get(target_key) or model_params.get(alt_key)

            if "raw" in parts:
                tensor = f.get_tensor(parts["raw"]).to(torch.bfloat16)
                raw_key = parts["raw"]
                alt_raw = raw_key.replace("language_model.", "", 1)
                param_r = model_params.get(raw_key) or model_params.get(alt_raw)
                if param_r is not None and tensor.shape == param_r.shape:
                    param_r.data = tensor.to(device)
                    loaded += 1
                else:
                    # Try as buffer
                    for name in [raw_key, alt_raw]:
                        parts_n = name.rsplit(".", 1)
                        if len(parts_n) == 2:
                            try:
                                parent = model
                                for p in parts_n[0].split("."):
                                    parent = getattr(parent, p)
                                if hasattr(parent, parts_n[1]):
                                    setattr(parent, parts_n[1], tensor.to(device))
                                    loaded += 1
                                    break
                            except (AttributeError, KeyError):
                                pass
                    else:
                        skipped += 1
            elif "weight" in parts and "scales" in parts:
                w = f.get_tensor(parts["weight"])
                s = f.get_tensor(parts["scales"])
                b = f.get_tensor(parts["biases"])
                dq = dequantize_4bit(w, s, b, GROUP_SIZE)

                if param is not None and dq.shape == param.shape:
                    param.data = dq.to(device)
                    loaded += 1
                else:
                    skipped += 1
                del dq
            else:
                skipped += 1

    vram1 = torch.cuda.memory_allocated() / 1e9
    print(f"  Loaded: {loaded}, Skipped: {skipped}")
    print(f"  VRAM: {vram1:.2f} GB [{time.time()-t0:.1f}s]")

    # ── Step 4: Set up sniper + patch MoE layers ──
    print("\n[3/5] Setting up Expert Sniper...")
    sniper = ExpertSniper(model_dir / "experts", num_layers, device=device, cache_layers=args.cache_layers)
    sniper.cache_in_vram()

    print("\n[4/5] Patching MoE layers...")
    patched = 0
    for i in range(num_layers):
        layer = model.model.layers[i]
        if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate') and layer.mlp.gate is not None:
            patch_moe_layer(layer.mlp, i, sniper, top_k)
            patched += 1
    print(f"  Patched: {patched}/{num_layers}")
    vram2 = torch.cuda.memory_allocated() / 1e9
    print(f"  VRAM after patch: {vram2:.2f} GB")

    # ── Step 5: Generate ──
    print("\n[5/5] Generating...")
    tokenizer = AutoTokenizer.from_pretrained(str(original_dir), trust_remote_code=True)

    messages = [
        {"role": "system", "content": "Answer briefly and directly."},
        {"role": "user", "content": args.prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    prompt_len = input_ids.shape[1]
    print(f"  Prompt: {prompt_len} tokens")

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )
    t_total = time.time() - t0

    new_tokens = output_ids[0][prompt_len:]
    n = len(new_tokens)
    tps = n / t_total if t_total > 0 else 0
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    vram_final = torch.cuda.memory_allocated() / 1e9

    print(f"\n{'='*60}")
    print(f"Q: {args.prompt}")
    print(f"A: {output_text}")
    print(f"{'='*60}")
    print(f"  Model: Qwen3.5-122B-A10B (69.6 GB)")
    print(f"  VRAM: {vram_final:.1f} GB")
    print(f"  Cached layers: 0-{args.cache_layers-1}")
    print(f"  Speed: {tps:.3f} tok/s")
    print(f"  Tokens: {n}")
    print(f"  Time: {t_total:.1f}s")


if __name__ == "__main__":
    main()
