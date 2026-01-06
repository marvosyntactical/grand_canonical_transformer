import os
import json
import math
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Your local module
from argos_viz import (
    capture_layernorm_flow,
    pca3_fit,
    pca3_transform,
    to_unit_sphere,
    color_gradient,
)

PRECOMPUTED_DIR = "precomputed"


# -----------------------------
# Pre-softmax score capture (robust for GPT2/DistilGPT2 style)
# -----------------------------

def patch_gpt2_style_attn_scores(model) -> Dict[str, Any]:
    patched_modules = 0
    for m in model.modules():
        if not hasattr(m, "_attn") or not callable(getattr(m, "_attn", None)):
            continue
        if hasattr(m, "_gce_is_patched") and m._gce_is_patched:
            continue

        orig__attn = m._attn

        def make_new__attn(attn_module, orig_fn):
            def new__attn(query, key, value, attention_mask=None, head_mask=None):
                scores = torch.matmul(query, key.transpose(-1, -2))
                if getattr(attn_module, "scale_attn_weights", False):
                    scores = scores / math.sqrt(value.size(-1))
                if hasattr(attn_module, "bias") and attn_module.bias is not None:
                    causal_mask = attn_module.bias[:, :, : scores.size(-2), : scores.size(-1)]
                    scores = torch.where(causal_mask, scores, torch.full_like(scores, -1e4))
                if attention_mask is not None:
                    scores = scores + attention_mask
                attn_module._gce_scores = scores.detach()
                return orig_fn(query, key, value, attention_mask=attention_mask, head_mask=head_mask)
            return new__attn

        m._attn = make_new__attn(m, orig__attn)
        m._gce_is_patched = True
        patched_modules += 1

    return {"patched": patched_modules > 0, "method": "gpt2_style__attn_patch", "patched_modules": patched_modules}


def install_score_capture(model) -> Dict[str, Any]:
    return patch_gpt2_style_attn_scores(model)


def collect_patched_scores(model) -> List[torch.Tensor]:
    scores = []
    for m in model.modules():
        if hasattr(m, "_gce_scores") and isinstance(m._gce_scores, torch.Tensor):
            scores.append(m._gce_scores)
    return scores


# -----------------------------
# Cache helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def seq_hash(model_name: str, ids: torch.Tensor) -> str:
    # ids: [T]
    h = hashlib.sha1()
    h.update(model_name.encode("utf-8"))
    h.update(b"\0")
    h.update(np.asarray(ids.detach().cpu(), dtype=np.int64).tobytes())
    return h.hexdigest()[:16]

def save_npz_cache(
    path: str,
    model_name: str,
    full_ids: torch.Tensor,       # [T]
    input_len: int,
    new_ids: torch.Tensor,        # [T_new]
    attentions: List[torch.Tensor],   # list of [1,H,T,T]
    scores: Optional[List[torch.Tensor]],  # list of [1,H,T,T] or None
    pca_payload: Dict[str, Any],       # includes X3, tokens, colors, eigvals_by_ln
    meta: Dict[str, Any],
) -> None:
    data: Dict[str, Any] = {}
    data["model_name"] = np.array(model_name)
    data["full_ids"] = np.asarray(full_ids.detach().cpu(), dtype=np.int64)
    data["input_len"] = np.array(int(input_len), dtype=np.int64)
    data["new_ids"] = np.asarray(new_ids.detach().cpu(), dtype=np.int64)

    data["L_attn"] = np.array(len(attentions), dtype=np.int64)
    for i, a in enumerate(attentions):
        data[f"attn_{i}"] = a.detach().cpu().to(torch.float16).numpy()

    if scores is None:
        data["L_scores"] = np.array(0, dtype=np.int64)
    else:
        data["L_scores"] = np.array(len(scores), dtype=np.int64)
        for i, s in enumerate(scores):
            data[f"score_{i}"] = s.detach().cpu().to(torch.float16).numpy()

    # PCA payload
    if pca_payload and "X3" in pca_payload:
        data["pca_X3"] = np.asarray(pca_payload["X3"], dtype=np.float16)  # (T, Lnorm, 3)
        data["pca_Lnorm"] = np.array(int(pca_payload.get("Lnorm", pca_payload["X3"].shape[1])), dtype=np.int64)
        # eigvals_by_ln is ragged: store as JSON + a packed array list
        eigvals_by_ln = pca_payload.get("eigvals_by_ln", [])
        data["pca_eigvals_json"] = np.array(json.dumps([ev.tolist() for ev in eigvals_by_ln]))
        data["pca_tokens_json"] = np.array(json.dumps(pca_payload.get("tokens", [])))
        data["pca_colors_json"] = np.array(json.dumps(pca_payload.get("colors", [])))
    else:
        data["pca_Lnorm"] = np.array(-1, dtype=np.int64)
        data["pca_tokens_json"] = np.array(json.dumps([]))
        data["pca_colors_json"] = np.array(json.dumps([]))
        data["pca_eigvals_json"] = np.array(json.dumps([]))

    data["meta_json"] = np.array(json.dumps(meta))

    np.savez_compressed(path, **data)

def build_default_meta() -> Dict[str, Any]:
    return {
        "created_at_unix": time.time(),
        "note": "precompute.py default run",
    }


# -----------------------------
# Main precompute run
# -----------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("HF_MODEL", "distilgpt2"))
    parser.add_argument("--prompt", default="Hey, how are ya?")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.dtype == "auto":
        dtype = torch.float32 if device == "cpu" else torch.float16
    else:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Force eager so attentions exist
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"

    model.to(device)
    model.eval()

    install_score_capture(model)

    # Deterministic-ish generation
    torch.manual_seed(args.seed)
    prompt_text = args.prompt.strip() + "\n"
    inputs = tok(prompt_text, return_tensors="pt", padding=False).to(device)
    input_len = int(inputs["input_ids"].shape[-1])

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=True,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            pad_token_id=tok.eos_token_id,
            return_dict_in_generate=True,
        )

    full_ids = gen.sequences[0]  # [T]
    new_ids = full_ids[input_len:]

    assistant_text = tok.decode(new_ids, skip_special_tokens=True).strip()
    print("Assistant reply:", assistant_text)

    diag_input_ids = full_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(diag_input_ids)

    # Clear old scores
    for m in model.modules():
        if hasattr(m, "_gce_scores"):
            delattr(m, "_gce_scores")

    with torch.no_grad():
        out = model(
            input_ids=diag_input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = list(out.attentions) if out.attentions is not None else []
    attentions = [a for a in attentions if a is not None]

    scores = collect_patched_scores(model)
    scores = scores if (len(scores) >= len(attentions) and len(attentions) > 0) else None

    # PCA payload
    pca_payload: Dict[str, Any] = {}
    try:
        Z = capture_layernorm_flow(model, diag_input_ids, attention_mask=attention_mask)  # (T, Lnorm, D)
        Tlen, Lnorm, D = Z.shape
        Z_flat = Z.reshape(-1, D)
        pca_state = pca3_fit(Z_flat)
        X3 = pca3_transform(Z_flat, pca_state)
        X3 = to_unit_sphere(X3).reshape(Tlen, Lnorm, 3)

        # eig spectra per LN
        eigvals_by_ln = []
        for ell in range(Lnorm):
            X = Z[:, ell, :]
            X = X - X.mean(axis=0, keepdims=True)
            _, S_svd, _ = np.linalg.svd(X, full_matrices=False)
            denom = max(Tlen - 1, 1)
            eigvals = (S_svd ** 2) / denom
            eigvals_by_ln.append(eigvals.astype(np.float64))

        tokens = tok.convert_ids_to_tokens(diag_input_ids[0].tolist())
        colors = color_gradient(len(tokens))

        pca_payload = {
            "X3": X3,
            "tokens": tokens,
            "colors": colors,
            "Lnorm": int(Lnorm),
            "eigvals_by_ln": eigvals_by_ln,
        }
    except Exception as e:
        pca_payload = {"error": str(e)}

    # Save cache
    ensure_dir(PRECOMPUTED_DIR)
    ensure_dir(os.path.join(PRECOMPUTED_DIR, args.model))

    h = seq_hash(args.model, full_ids)
    out_path = os.path.join(PRECOMPUTED_DIR, args.model, f"{h}.npz")

    meta = build_default_meta()
    meta.update(
        {
            "prompt": args.prompt,
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "seed": int(args.seed),
            "dtype": str(args.dtype),
            "device": str(device),
            "transformers_version": __import__("transformers").__version__,
            "assistant_text": assistant_text,
        }
    )

    save_npz_cache(
        out_path,
        model_name=args.model,
        full_ids=full_ids,
        input_len=input_len,
        new_ids=new_ids,
        attentions=attentions,
        scores=scores,
        pca_payload=pca_payload if "error" not in pca_payload else {},
        meta=meta,
    )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

