import os
import json
import math
import time
import hashlib
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from argos_viz import (
    capture_layernorm_flow,
    pca3_fit,
    pca3_transform,
    to_unit_sphere,
    color_gradient,
)

PRECOMPUTED_DIR = "precomputed"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------
# Robust patching utilities
# ---------------------------

def _filter_kwargs_for(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filter kwargs to only those accepted by fn's signature (prevents unexpected kwarg crashes)."""
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        # If fn accepts **kwargs, keep everything
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        # If signature introspection fails, best effort: keep kwargs
        return kwargs


def _try_patch__attn_method(model) -> Dict[str, Any]:
    patched = 0
    for m in model.modules():
        if not hasattr(m, "_attn") or not callable(getattr(m, "_attn")):
            continue
        if getattr(m, "_gce_is_patched", False):
            continue

        orig = m._attn

        def make_new(attn_module, orig_fn):
            def new__attn(query, key, value, attention_mask=None, head_mask=None, **kwargs):
                scores = torch.matmul(query, key.transpose(-1, -2))  # [B,H,T,T]
                if getattr(attn_module, "scale_attn_weights", False):
                    scores = scores / math.sqrt(value.size(-1))

                if hasattr(attn_module, "bias") and attn_module.bias is not None:
                    causal_mask = attn_module.bias[:, :, : scores.size(-2), : scores.size(-1)]
                    scores = torch.where(causal_mask, scores, torch.full_like(scores, -1e4))

                if attention_mask is not None:
                    scores = scores + attention_mask

                attn_module._gce_scores = scores.detach()
                kw = _filter_kwargs_for(orig_fn, dict(kwargs))
                return orig_fn(query, key, value, attention_mask=attention_mask, head_mask=head_mask, **kw)

            return new__attn

        m._attn = make_new(m, orig)
        m._gce_is_patched = True
        patched += 1

    return {"patched": patched > 0, "method": "patch__attn", "patched_modules": patched}


def _try_patch_gpt2attention_forward(model) -> Dict[str, Any]:
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    except Exception as e:
        return {"patched": False, "method": "patch_gpt2_forward", "patched_modules": 0, "error": str(e)}

    patched = 0
    for m in model.modules():
        if not isinstance(m, GPT2Attention):
            continue
        if getattr(m, "_gce_is_patched", False):
            continue

        orig_forward = m.forward

        def make_new_forward(attn_module, orig_fwd):
            def new_forward(*args, **kwargs):
                # Support both old and new naming for cached keys/values
                if "past_key_values" in kwargs and "layer_past" not in kwargs:
                    kwargs["layer_past"] = kwargs["past_key_values"]

                # hidden_states is first positional arg after self in GPT2Attention.forward
                hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states", None)
                if hidden_states is None:
                    # fall back to original (should not happen)
                    kw = _filter_kwargs_for(orig_fwd, dict(kwargs))
                    return orig_fwd(*args, **kw)

                layer_past = kwargs.get("layer_past", None)
                attention_mask = kwargs.get("attention_mask", None)

                with torch.no_grad():
                    qkv = attn_module.c_attn(hidden_states)
                    query, key, value = qkv.split(attn_module.split_size, dim=2)

                    query = attn_module._split_heads(query, attn_module.num_heads, attn_module.head_dim)
                    key = attn_module._split_heads(key, attn_module.num_heads, attn_module.head_dim)
                    value = attn_module._split_heads(value, attn_module.num_heads, attn_module.head_dim)

                    if layer_past is not None:
                        past_key, past_value = layer_past
                        key = torch.cat((past_key, key), dim=-2)
                        value = torch.cat((past_value, value), dim=-2)

                    scores = torch.matmul(query, key.transpose(-1, -2))
                    scores = scores / math.sqrt(value.size(-1))

                    if hasattr(attn_module, "bias") and attn_module.bias is not None:
                        qlen = scores.size(-2)
                        klen = scores.size(-1)
                        causal_mask = attn_module.bias[:, :, klen - qlen : klen, :klen]
                        scores = torch.where(causal_mask, scores, torch.full_like(scores, -1e4))

                    if attention_mask is not None:
                        scores = scores + attention_mask

                    attn_module._gce_scores = scores.detach()

                # Don't forward unexpected kwargs to the original implementation.
                # In particular, remove past_key_values if orig doesn't accept it.
                cleaned = dict(kwargs)
                if "past_key_values" in cleaned:
                    # keep it only if accepted
                    pass
                kw = _filter_kwargs_for(orig_fwd, cleaned)
                return orig_fwd(*args, **kw)

            return new_forward

        m.forward = make_new_forward(m, orig_forward)
        m._gce_is_patched = True
        patched += 1

    return {"patched": patched > 0, "method": "patch_gpt2_forward", "patched_modules": patched}


def install_score_capture(model) -> Dict[str, Any]:
    info = _try_patch__attn_method(model)
    if info.get("patched"):
        return info
    info2 = _try_patch_gpt2attention_forward(model)
    return info2


def collect_scores(model) -> List[torch.Tensor]:
    out = []
    for m in model.modules():
        if hasattr(m, "_gce_scores") and isinstance(m._gce_scores, torch.Tensor):
            out.append(m._gce_scores)
    return out


# ---------------------------
# Cache naming & IO
# ---------------------------

def seq_hash(model_name: str, ids: torch.Tensor) -> str:
    h = hashlib.sha1()
    h.update(model_name.encode("utf-8"))
    h.update(b"\0")
    h.update(np.asarray(ids.detach().cpu(), dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def save_npz_cache(
    path: str,
    model_name: str,
    full_ids: torch.Tensor,
    input_len: int,
    new_ids: torch.Tensor,
    attentions: List[torch.Tensor],
    scores: Optional[List[torch.Tensor]],
    pca_payload: Dict[str, Any],
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

    if pca_payload and "X3" in pca_payload:
        data["pca_X3"] = np.asarray(pca_payload["X3"], dtype=np.float16)
        data["pca_Lnorm"] = np.array(int(pca_payload["X3"].shape[1]), dtype=np.int64)
        data["pca_tokens_json"] = np.array(json.dumps(pca_payload.get("tokens", [])))
        data["pca_colors_json"] = np.array(json.dumps(pca_payload.get("colors", [])))
        eigvals_by_ln = pca_payload.get("eigvals_by_ln", [])
        data["pca_eigvals_json"] = np.array(json.dumps([ev.tolist() for ev in eigvals_by_ln]))
    else:
        data["pca_Lnorm"] = np.array(-1, dtype=np.int64)
        data["pca_tokens_json"] = np.array(json.dumps([]))
        data["pca_colors_json"] = np.array(json.dumps([]))
        data["pca_eigvals_json"] = np.array(json.dumps([]))

    data["meta_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **data)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("HF_MODEL", "distilgpt2"))
    parser.add_argument("--prompt", default="Hey, how are ya?")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
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

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"

    model.to(device)
    model.eval()

    patch_info = install_score_capture(model)

    torch.manual_seed(args.seed)

    prompt_text = args.prompt.strip() + "\n"
    inputs = tok(prompt_text, return_tensors="pt").to(device)
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

    full_ids = gen.sequences[0]
    new_ids = full_ids[input_len:]
    assistant_text = tok.decode(new_ids, skip_special_tokens=True).strip()

    # Now do a full forward pass for attentions + scores
    diag_input_ids = full_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(diag_input_ids)

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

    attentions = [a for a in (out.attentions or []) if a is not None]
    scores = collect_scores(model)
    scores = scores if (len(attentions) > 0 and len(scores) >= len(attentions)) else None

    # PCA payload
    pca_payload: Dict[str, Any] = {}
    try:
        Z = capture_layernorm_flow(model, diag_input_ids, attention_mask=attention_mask)  # (T,Lnorm,D)
        Tlen, Lnorm, D = Z.shape
        Z_flat = Z.reshape(-1, D)
        pca_state = pca3_fit(Z_flat)
        X3 = pca3_transform(Z_flat, pca_state)
        X3 = to_unit_sphere(X3).reshape(Tlen, Lnorm, 3)

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
            "eigvals_by_ln": eigvals_by_ln,
        }
    except Exception as e:
        pca_payload = {"error": str(e)}

    ensure_dir(PRECOMPUTED_DIR)
    ensure_dir(os.path.join(PRECOMPUTED_DIR, args.model))

    h = seq_hash(args.model, full_ids)
    out_path = os.path.join(PRECOMPUTED_DIR, args.model, f"{h}.npz")

    meta = {
        "created_at_unix": time.time(),
        "prompt": args.prompt,
        "assistant_text": assistant_text,
        "max_new_tokens": int(args.max_new_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "seed": int(args.seed),
        "dtype": str(args.dtype),
        "device": str(device),
        "transformers_version": __import__("transformers").__version__,
        "score_capture": patch_info,
    }

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
    print(f"Score capture: {patch_info}")
    print(f"Scores captured: {0 if scores is None else len(scores)}; attentions: {len(attentions)}")


if __name__ == "__main__":
    main()

