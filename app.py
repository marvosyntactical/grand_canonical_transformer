import math
import os
import json
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from argos_viz import (
    capture_layernorm_flow,
    pca3_fit,
    pca3_transform,
    to_unit_sphere,
    build_layer_slider_figure,
    color_gradient,
)

PRECOMPUTED_DIR = "precomputed"


# ---------------------------
# Helpers: kwargs filtering
# ---------------------------

def _filter_kwargs_for(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


# ---------------------------
# Numerics
# ---------------------------

def safe_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.float().clamp(eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)

def topk_mass(p: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), p.shape[-1])
    vals, _ = torch.topk(p.float(), k=k, dim=-1)
    return vals.sum(dim=-1)

def softmax_beta(scores: torch.Tensor, beta: float) -> torch.Tensor:
    return torch.softmax((beta * scores).float(), dim=-1)

def var_under(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    p = p.float()
    x = x.float()
    m = (p * x).sum(dim=-1)
    m2 = (p * (x * x)).sum(dim=-1)
    return m2 - m * m


# ---------------------------
# Score capture
# ---------------------------

def _filter_kwargs_for(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return kwargs
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs


def _split_heads_fallback(attn_module, x: torch.Tensor) -> torch.Tensor:
    if hasattr(attn_module, "_split_heads") and callable(getattr(attn_module, "_split_heads")):
        return attn_module._split_heads(x, attn_module.num_heads, attn_module.head_dim)
    if hasattr(attn_module, "split_heads") and callable(getattr(attn_module, "split_heads")):
        return attn_module.split_heads(x)

    B, T, D = x.shape
    n_head = getattr(attn_module, "num_heads", None)
    if n_head is None:
        n_head = getattr(attn_module, "n_head", None)
    if n_head is None:
        raise AttributeError("Cannot infer num_heads / n_head for GPT2Attention split.")

    head_dim = getattr(attn_module, "head_dim", None)
    if head_dim is None:
        head_dim = getattr(attn_module, "head_size", None)
    if head_dim is None:
        split_size = getattr(attn_module, "split_size", None)
        if split_size is not None:
            head_dim = int(split_size // n_head)
        else:
            head_dim = int(D // n_head)

    return x.view(B, T, n_head, head_dim).permute(0, 2, 1, 3).contiguous()


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
                scores = torch.matmul(query, key.transpose(-1, -2))
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
                if "past_key_values" in kwargs and "layer_past" not in kwargs:
                    kwargs["layer_past"] = kwargs["past_key_values"]

                hidden_states = args[0] if len(args) > 0 else kwargs.get("hidden_states", None)
                if hidden_states is None:
                    kw = _filter_kwargs_for(orig_fwd, dict(kwargs))
                    return orig_fwd(*args, **kw)

                layer_past = kwargs.get("layer_past", None)
                attention_mask = kwargs.get("attention_mask", None)

                with torch.no_grad():
                    qkv = attn_module.c_attn(hidden_states)
                    split_size = getattr(attn_module, "split_size", None)
                    if split_size is None:
                        query, key, value = qkv.chunk(3, dim=2)
                    else:
                        query, key, value = qkv.split(split_size, dim=2)

                    query = _split_heads_fallback(attn_module, query)
                    key = _split_heads_fallback(attn_module, key)
                    value = _split_heads_fallback(attn_module, value)

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

                cleaned = dict(kwargs)
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
    return _try_patch_gpt2attention_forward(model)


def collect_scores(model) -> List[torch.Tensor]:
    out = []
    for m in model.modules():
        if hasattr(m, "_gce_scores") and isinstance(m._gce_scores, torch.Tensor):
            out.append(m._gce_scores)
    return out

# ---------------------------
# Precomputed cache loading (read-only)
# ---------------------------

def find_latest_cache_for_model(model_name: str) -> Optional[str]:
    d = os.path.join(PRECOMPUTED_DIR, model_name)
    if not os.path.isdir(d):
        return None
    files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def load_npz_cache(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    model_name = str(z["model_name"])
    full_ids = torch.tensor(z["full_ids"].astype(np.int64))
    input_len = int(z["input_len"])
    new_ids = z["new_ids"].astype(np.int64)

    L_attn = int(z["L_attn"])
    attentions = [torch.tensor(z[f"attn_{i}"]) for i in range(L_attn)]

    L_scores = int(z["L_scores"])
    scores = None
    if L_scores > 0:
        scores = [torch.tensor(z[f"score_{i}"]) for i in range(L_scores)]

    pca_payload: Dict[str, Any] = {}
    pca_Lnorm = int(z["pca_Lnorm"])
    if pca_Lnorm >= 0 and "pca_X3" in z.files:
        X3 = z["pca_X3"].astype(np.float16)
        tokens = json.loads(str(z["pca_tokens_json"]))
        colors = json.loads(str(z["pca_colors_json"]))
        eigvals_by_ln = json.loads(str(z["pca_eigvals_json"]))
        pca_payload = {
            "X3": X3,
            "Lnorm": pca_Lnorm,
            "tokens": tokens,
            "colors": colors,
            "eigvals_by_ln": [np.array(ev, dtype=np.float64) for ev in eigvals_by_ln],
        }

    meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
    return {
        "cache_path": path,
        "model_name": model_name,
        "full_ids": full_ids,
        "input_len": input_len,
        "new_ids": new_ids,
        "attentions": attentions,
        "scores": scores,
        "pca": pca_payload,
        "meta": meta,
    }


# ---------------------------
# Plotly builders
# ---------------------------

def fig_line(x, ys: Dict[str, np.ndarray], title: str, x_title="Layer", y_title="Value", height=340):
    fig = go.Figure()
    for name, y in ys.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, height=height)
    return fig

def fig_spectrum(spec: np.ndarray, title: str):
    x = np.arange(1, len(spec) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=spec, mode="lines", name="sorted α"))
    fig.update_layout(title=title, xaxis_title="Rank", yaxis_title="Attention weight", height=340)
    fig.update_yaxes(type="log")
    return fig

def fig_surface(Z: np.ndarray, title: str, x_title: str, y_title: str, z_title: str):
    steps = np.arange(Z.shape[0])
    layers = np.arange(Z.shape[1])
    fig = go.Figure(data=[go.Surface(z=Z, x=layers, y=steps)])
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title=x_title, yaxis_title=y_title, zaxis_title=z_title),
        height=480,
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def fig_decomposition(phi, betaU, H, signed_resid, abs_resid, title: str):
    x = np.arange(len(phi))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=betaU, name="β·U"))
    fig.add_trace(go.Bar(x=x, y=H, name="H"))
    fig.add_trace(go.Scatter(x=x, y=phi, mode="lines+markers", name="φ = log∑exp(βs)"))
    fig.add_trace(go.Scatter(x=x, y=signed_resid, mode="lines+markers", name="r = φ-(βU+H)", yaxis="y2"))
    fig.add_trace(go.Scatter(x=x, y=abs_resid, mode="lines+markers", name="|r|", yaxis="y3"))

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title="Layer",
        yaxis_title="Value",
        yaxis2=dict(title="Signed residual r", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="Abs residual |r|", overlaying="y", side="right", showgrid=False, position=0.95, rangemode="tozero"),
        height=460,
        legend=dict(orientation="h"),
    )
    return fig


# ---------------------------
# Accessors
# ---------------------------

def get_attention_row(attn: torch.Tensor, token_index: int, head: Optional[int]) -> torch.Tensor:
    A = attn[0]  # [H,T,T]
    T = A.shape[-1]
    q = max(0, min(int(token_index), T - 1))
    row = A[:, q, :]  # [H,T]
    if head is None:
        return row.mean(dim=0)
    h = max(0, min(int(head), row.shape[0] - 1))
    return row[h]

def get_score_row(score: torch.Tensor, token_index: int, head: Optional[int]) -> torch.Tensor:
    S = score[0]
    T = S.shape[-1]
    q = max(0, min(int(token_index), T - 1))
    row = S[:, q, :]
    if head is None:
        return row.mean(dim=0)
    h = max(0, min(int(head), row.shape[0] - 1))
    return row[h]


# ---------------------------
# Prompting (simple + stable)
# ---------------------------

QA_FEWSHOT_PREFIX = """Q: Hey, how are you?
A: Pretty good — what’s up?

Q: Explain softmax in one sentence.
A: It turns scores into a probability distribution by exponentiating and normalizing.

"""

def build_prompt_plain(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages[1:]:
        parts.append(msg["content"].strip())
    return "\n\n".join([p for p in parts if p]).strip() + "\n"

def build_prompt_qa(messages: List[Dict[str, str]]) -> str:
    turns = []
    for msg in messages[1:]:
        if msg["role"] == "user":
            turns.append(("Q", msg["content"].strip()))
        elif msg["role"] == "assistant":
            turns.append(("A", msg["content"].strip()))
    lines = [QA_FEWSHOT_PREFIX]
    for tag, text in turns:
        if text:
            lines.append(f"{tag}: {text}\n")
    lines.append("A: ")
    return "".join(lines)


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="The Transformer as Grand Canonical Ensemble", layout="wide")
st.title("The Transformer as Grand Canonical Ensemble")
st.caption("Chat with a HuggingFace causal LM and visualize GCE-style diagnostics + post-LN latent geometry.")

# icon-y primary button
st.markdown(
    """
<style>
button[kind="primary"] {
  border-radius: 999px !important;
  height: 2.6rem !important;
  width: 2.6rem !important;
  padding: 0 !important;
  font-size: 1.25rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model")
    model_name = st.text_input("HF model name", value=os.environ.get("HF_MODEL", "distilgpt2"))
    device_pref = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    dtype_pref = st.selectbox("dtype", ["auto", "float16", "bfloat16", "float32"], index=0)

    st.header("Generation")
    prompt_style = st.selectbox("Prompt style", ["Q/A few-shot (recommended for distilgpt2)", "Plain continuation"], index=1)
    max_new_tokens = st.slider("max_new_tokens", 1, 256, 64, 1)
    temperature = st.slider("temperature", 0.1, 2.0, 0.8, 0.1)
    top_p = st.slider("top_p", 0.1, 1.0, 0.95, 0.05)

    st.header("GCE parameters")
    beta = st.slider("β (for αβ=softmax(βs) in thermo-consistent plots)", 0.1, 10.0, 1.0, 0.1)

    st.header("Visualization")
    head_mode = st.selectbox("Head selection", ["mean over heads", "specific head"], index=0)
    head_index = st.number_input("Head index", value=0, step=1)
    spectrum_layer = st.number_input("Spectrum layer", value=0, step=1)
    spectrum_head = st.number_input("Spectrum head", value=0, step=1)
    show_3d = st.checkbox("Show 3D surfaces (step × layer)", value=True)

    st.header("Latent PCA")
    show_latent_pca = st.checkbox("Show post-LN PCA (3D)", value=True)
    show_trajectories = st.checkbox("Show token trajectories", value=True)

    st.header("Diagnostics mode")
    decomp_mode = st.selectbox(
        "Decomposition uses",
        [
            "Thermo-consistent: αβ = softmax(β·scores) (identity should close)",
            "Empirical: use model attention α (identity may NOT close)",
        ],
        index=0,
    )

    st.header("Chat controls")
    clear = st.button("Clear chat", use_container_width=True)


@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer(model_name: str, device_pref: str, dtype_pref: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if device_pref == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_pref

    if dtype_pref == "auto":
        dtype = torch.float32 if device == "cpu" else torch.float16
    else:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_pref]

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"

    model.to(device)
    model.eval()

    patch_info = install_score_capture(model)
    return tok, model, device, dtype, patch_info

tokenizer, model, device, dtype, patch_info = load_model_and_tokenizer(model_name, device_pref, dtype_pref)
st.sidebar.caption(f"Score capture: {patch_info}")


# =========================
# Seed load
# =========================

def reset_to_seed(seed: Dict[str, Any]) -> None:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if seed.get("prompt"):
        st.session_state.messages.append({"role": "user", "content": seed["prompt"]})
    if seed.get("assistant_text"):
        st.session_state.messages.append({"role": "assistant", "content": seed["assistant_text"]})
    st.session_state.diag = seed.get("diag", {})
    st.session_state.has_seed = True

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "diag" not in st.session_state:
    st.session_state.diag = {}
if "seed" not in st.session_state:
    st.session_state.seed = {}
if "has_seed" not in st.session_state:
    st.session_state.has_seed = False

if not st.session_state.has_seed:
    latest = find_latest_cache_for_model(model_name)
    if latest is not None:
        cached = load_npz_cache(latest)
        meta = cached.get("meta", {}) or {}
        prompt = meta.get("prompt", "Hey, how are ya?")
        assistant_text = meta.get("assistant_text", "")

        pca_payload = cached.get("pca", {})
        if pca_payload and "X3" in pca_payload:
            fig_pca = build_layer_slider_figure(
                pca_payload["X3"],
                token_text=pca_payload.get("tokens", []),
                show_trajectories=bool(show_trajectories),
                token_colors=pca_payload.get("colors", None),
                title="Post-LayerNorm token geometry (PCA3 on unit sphere)",
            )
            pca_payload["fig"] = fig_pca

        seed_diag = {
            "input_len": cached["input_len"],
            "new_ids": cached["new_ids"],
            "attentions": cached["attentions"],
            "scores": cached["scores"],
            "diag_input_ids": cached["full_ids"].unsqueeze(0).cpu(),
            "pca": pca_payload,
            "_seed_cache": latest,
            "_seed_meta": meta,
        }
        st.session_state.seed = {"prompt": prompt, "assistant_text": assistant_text, "diag": seed_diag}
    else:
        st.session_state.seed = {"prompt": "Hey, how are ya?", "assistant_text": "", "diag": {}}

    reset_to_seed(st.session_state.seed)

if clear:
    reset_to_seed(st.session_state.seed)
    st.rerun()


# =========================
# Diagnostics compute
# =========================

def compute_diagnostics_for_sequence(full_ids_1d: torch.Tensor, input_len: int) -> Dict[str, Any]:
    diag_input_ids = full_ids_1d.unsqueeze(0).to(device)
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

    pca_payload: Dict[str, Any] = {}
    if show_latent_pca:
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

            tok_text = tokenizer.convert_ids_to_tokens(diag_input_ids[0].tolist())
            tok_colors = color_gradient(len(tok_text))

            fig_pca = build_layer_slider_figure(
                X3,
                token_text=tok_text,
                show_trajectories=bool(show_trajectories),
                token_colors=tok_colors,
                title="Post-LayerNorm token geometry (PCA3 on unit sphere)",
            )

            pca_payload = {
                "X3": X3.astype(np.float16),
                "tokens": tok_text,
                "colors": tok_colors,
                "Lnorm": int(Lnorm),
                "eigvals_by_ln": eigvals_by_ln,
                "fig": fig_pca,
            }
        except Exception as e:
            pca_payload = {"error": str(e)}

    return {
        "input_len": int(input_len),
        "new_ids": full_ids_1d[input_len:].detach().cpu().numpy().astype(np.int64),
        "attentions": attentions,
        "scores": scores,
        "diag_input_ids": full_ids_1d.unsqueeze(0).detach().cpu(),
        "pca": pca_payload,
    }


# =========================
# Chat rendering (input LAST)
# =========================

with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_input("Message", value="", placeholder="Type a message…", label_visibility="collapsed", width=1000)
    submitted = st.form_submit_button("➤", type="primary")

if submitted:
    if not user_text.strip():
        st.warning("Type a message first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_text.strip()})
        with st.chat_message("assistant"):
            status = st.status("Generating…", expanded=True)
            try:
                # status.write("Building prompt…")
                prompt_text = build_prompt_qa(st.session_state.messages) if prompt_style.startswith("Q/A") else build_prompt_plain(st.session_state.messages)

                # status.write("Tokenizing…")
                inputs = tokenizer(prompt_text, return_tensors="pt", padding=False).to(device)
                input_len = int(inputs["input_ids"].shape[-1])

                status.write("Sampling completion…")
                with torch.no_grad():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=int(max_new_tokens),
                        do_sample=True,
                        temperature=float(temperature),
                        top_p=float(top_p),
                        pad_token_id=tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                    )

                full_ids = gen.sequences[0]
                new_ids = full_ids[input_len:]
                assistant_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                if assistant_text.startswith("A:"):
                    assistant_text = assistant_text[2:].strip()

                status.write("Computing diagnostics…")
                diag = compute_diagnostics_for_sequence(full_ids, input_len)

                status.update(label="Done", state="complete")
                # st.markdown(assistant_text if assistant_text else "*[empty completion]*")

                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                st.session_state.diag = diag

            except Exception as e:
                status.update(label="Generation failed", state="error")
                st.exception(e)

st.markdown("### Chat")
for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])



# =========================
# Diagnostics
# =========================

st.markdown("---")
st.subheader("Diagnostics")

with st.expander("❓ Global explainer: what is valid, and when?", expanded=False):
    st.markdown("We treat a single attention row as an ensemble over key positions.")
    st.markdown("**Two different attention distributions:**")
    st.markdown("- **Empirical**: the model’s actual attention weights.")
    st.markdown("- **Thermo-consistent**: build $\\alpha_\\beta = \\mathrm{softmax}(\\beta s)$ from captured pre-softmax scores $s$.")

    st.markdown("**Exact identity (only for $\\alpha_\\beta = \\mathrm{softmax}(\\beta s)$):**")
    st.latex(r"\phi(\beta) := \log\sum_j e^{\beta s_j}")
    st.latex(r"U(\beta) := \langle s, \alpha_\beta\rangle")
    st.latex(r"H(\alpha_\beta) := -\sum_j \alpha_{\beta,j}\log \alpha_{\beta,j}")
    st.latex(r"\phi(\beta) = \beta U(\beta) + H(\alpha_\beta)")
    st.markdown("Compute this **per head**, then average scalars. Don’t average scores/alphas first.")

diag = st.session_state.diag
if not diag or not diag.get("attentions"):
    st.info("No diagnostics yet (run precompute.py or send a chat message).")
    st.stop()

attentions = diag["attentions"]
scores = diag.get("scores", None)
input_len = int(diag["input_len"])
new_ids = np.asarray(diag["new_ids"], dtype=np.int64)
steps = int(len(new_ids))
L = len(attentions)

with st.sidebar:
    st.header("Forward-pass step")
    if steps > 0:
        step_idx = st.slider("Generated token step t", 0, steps - 1, steps - 1, 1)
        tok_str = tokenizer.decode([int(new_ids[step_idx])], skip_special_tokens=False)
        st.caption(f"Selected token: `{tok_str}` (id={int(new_ids[step_idx])})")
    else:
        step_idx = 0
        st.caption("No generated tokens in this run.")

chosen_head = None if head_mode == "mean over heads" else int(head_index)
token_index_abs = int(input_len + step_idx)
layers = np.arange(L)

# Plot 1: attention stats
with st.expander("ℹ️ Plot: attention ensemble stats", expanded=False):
    st.markdown("Computed from empirical attention weights $\\alpha$ for the selected token.")
    st.latex(r"H(\alpha)=-\sum_j \alpha_j\log \alpha_j,\quad N_{\mathrm{eff}}=\exp(H)")

H_list, Neff_list, top1_list, top5_list = [], [], [], []
for l in range(L):
    p = get_attention_row(attentions[l], token_index_abs, chosen_head)
    H = float(safe_entropy(p).item())
    H_list.append(H)
    Neff_list.append(float(math.exp(H)))
    top1_list.append(float(torch.max(p.float()).item()))
    top5_list.append(float(topk_mass(p, 5).item()))

st.plotly_chart(
    fig_line(
        layers,
        {
            "H_att": np.array(H_list),
            "N_eff=exp(H)": np.array(Neff_list),
            "top1 mass": np.array(top1_list),
            "top5 mass": np.array(top5_list),
        },
        title=f"Attention ensemble stats vs layer (step t={step_idx})",
    ),
    use_container_width=True,
)

# Plot 2: phase plot
with st.expander("ℹ️ Plot: phase plot (U vs H)", expanded=False):
    st.markdown("Each point is a layer mapped to $(H, U)$.")
    st.latex(r"U=\langle s,\alpha\rangle=\sum_j \alpha_j s_j")
    st.markdown("Use thermo-consistent mode to make $(H,U)$ reflect the canonical-family induced by $s$.")

if scores is None:
    st.warning("No pre-softmax scores captured → U/φ/Ω plots unavailable. (Patch should now work; if it says patched=True but still missing scores, tell me your transformers version.)")
else:
    U_list, H_for_phase = [], []
    for l in range(L):
        srow = get_score_row(scores[l], token_index_abs, chosen_head)
        if decomp_mode.startswith("Thermo-consistent"):
            a = softmax_beta(srow, beta=float(beta))
            H_for_phase.append(float(safe_entropy(a).item()))
            U_list.append(float((a.float() * srow.float()).sum().item()))
        else:
            p = get_attention_row(attentions[l], token_index_abs, chosen_head)
            H_for_phase.append(float(safe_entropy(p).item()))
            U_list.append(float((p.float() * srow.float()).sum().item()))

    fig_phase = go.Figure()
    fig_phase.add_trace(go.Scatter(x=np.array(H_for_phase), y=np.array(U_list), mode="lines+markers", name="layers"))
    fig_phase.update_layout(title=f"Phase plot (step t={step_idx}): U vs H", xaxis_title="Entropy H", yaxis_title="U = ⟨s,α⟩", height=340)
    st.plotly_chart(fig_phase, use_container_width=True)

# Plot 3: occupation spectrum
with st.expander("ℹ️ Plot: occupation spectrum", expanded=False):
    st.markdown("Sorted attention weights $\\alpha_{(1)}\\ge\\alpha_{(2)}\\ge\\dots$ for a chosen layer/head.")

spec_layer = int(spectrum_layer)
spec_head = int(spectrum_head)
A = attentions[spec_layer][0]
T = A.shape[-1]
q = max(0, min(int(token_index_abs), T - 1))
h = max(0, min(int(spec_head), A.shape[0] - 1))
p = A[h, q, :].float().cpu().numpy()
spec = np.sort(p)[::-1]
st.plotly_chart(fig_spectrum(spec, title=f"Occupation spectrum — step={step_idx}, layer={spec_layer}, head={spec_head}"), use_container_width=True)

# Plot 4: decomposition
with st.expander("ℹ️ Plot: energy–entropy decomposition + residuals", expanded=False):
    st.markdown("Exact identity for thermo-consistent $\\alpha_\\beta=\\mathrm{softmax}(\\beta s)$:")
    st.latex(r"\phi=\log\sum_j e^{\beta s_j}=\beta U+H,\quad \Omega=-\phi/\beta")
    st.markdown("We plot signed residual $r$ and absolute residual $|r|$ separately.")

if scores is not None:
    phi, betaU, Hdec, signed_r, abs_r = [], [], [], [], []
    for l in range(L):
        srow = get_score_row(scores[l], token_index_abs, chosen_head).float()
        if decomp_mode.startswith("Thermo-consistent"):
            a = softmax_beta(srow, beta=float(beta))
        else:
            a = get_attention_row(attentions[l], token_index_abs, chosen_head).float()

        ph = float(torch.logsumexp(float(beta) * srow, dim=-1).item())
        U = float((a * srow).sum().item())
        Hh = float(safe_entropy(a).item())

        r = ph - (float(beta) * U + Hh)
        phi.append(ph)
        betaU.append(float(beta) * U)
        Hdec.append(Hh)
        signed_r.append(r)
        abs_r.append(abs(r))

    st.plotly_chart(
        fig_decomposition(
            np.array(phi), np.array(betaU), np.array(Hdec),
            np.array(signed_r), np.array(abs_r),
            title=f"Energy–entropy decomposition (step={step_idx})",
        ),
        use_container_width=True,
    )

# Step×Layer surfaces (H, U, Ω)
if show_3d and steps > 0:
    with st.expander("ℹ️ Plot: step×layer surfaces", expanded=False):
        st.markdown("Surfaces show how a quantity evolves across generated step $t$ and layer $\\ell$.")

    H_mat = np.zeros((steps, L), dtype=float)
    for t in range(steps):
        idx = int(input_len + t)
        for l in range(L):
            p = get_attention_row(attentions[l], idx, chosen_head)
            H_mat[t, l] = float(safe_entropy(p).item())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(fig_surface(H_mat, "Entropy surface H(step, layer) [empirical α]", "Layer", "Generated step t", "H"), use_container_width=True)

    if scores is not None:
        U_mat = np.zeros((steps, L), dtype=float)
        Om_mat = np.zeros((steps, L), dtype=float)
        for t in range(steps):
            idx = int(input_len + t)
            for l in range(L):
                srow = get_score_row(scores[l], idx, chosen_head).float()
                a = softmax_beta(srow, beta=float(beta))
                U = float((a * srow).sum().item())
                ph = float(torch.logsumexp(float(beta) * srow, dim=-1).item())
                U_mat[t, l] = U
                Om_mat[t, l] = -ph / float(beta)

        with c2:
            st.plotly_chart(fig_surface(U_mat, "U surface ⟨s,αβ⟩(step, layer) [thermo-consistent]", "Layer", "Generated step t", "U"), use_container_width=True)
        with c3:
            st.plotly_chart(fig_surface(Om_mat, "Ω surface (grand potential proxy) Ω(step, layer)", "Layer", "Generated step t", "Ω"), use_container_width=True)
    else:
        with c2:
            st.info("U surface requires pre-softmax scores.")
        with c3:
            st.info("Ω surface requires pre-softmax scores.")

# Latent PCA + spectrum
if show_latent_pca:
    st.markdown("---")
    st.subheader("Post-LayerNorm PCA (3D) + Full spectrum")

    with st.expander("ℹ️ Plot: post-LN PCA + eigenvalue spectrum", expanded=False):
        st.markdown("We capture token states after each LayerNorm, fit PCA(3) globally, then project.")
        st.markdown("Right: full eigenvalue spectrum for the selected LN index (log-y).")

    pca = diag.get("pca", {})
    if not pca:
        st.info("No PCA computed for this run.")
    elif "error" in pca:
        st.warning(f"PCA capture failed: {pca['error']}")
    else:
        Lnorm = int(pca.get("Lnorm", 0))
        ln_sel = st.slider("LayerNorm index", 0, max(0, Lnorm - 1), 0, 1)

        left, right = st.columns(2, gap="large")
        with left:
            fig = pca.get("fig", None)
            if fig is None and "X3" in pca:
                fig = build_layer_slider_figure(
                    pca["X3"],
                    token_text=pca.get("tokens", []),
                    show_trajectories=bool(show_trajectories),
                    token_colors=pca.get("colors", None),
                    title="Post-LayerNorm token geometry (PCA3 on unit sphere)",
                )
                pca["fig"] = fig
            if fig.layout.sliders and len(fig.layout.sliders) > 0:
                fig.layout.sliders[0].active = int(ln_sel)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            eigvals_by_ln = pca.get("eigvals_by_ln", None)
            if eigvals_by_ln is None or len(eigvals_by_ln) == 0:
                st.info("Eigenvalue spectrum unavailable.")
            else:
                ev = np.array(eigvals_by_ln[int(ln_sel)], dtype=float)
                ev = np.sort(ev)[::-1]
                x = np.arange(1, len(ev) + 1)
                fig_ev = go.Figure()
                fig_ev.add_trace(go.Scatter(x=x, y=ev, mode="lines", name="eigenvalues"))
                fig_ev.update_layout(title=f"Eigenvalue spectrum (LayerNorm {ln_sel})", xaxis_title="Rank", yaxis_title="Eigenvalue", height=520)
                fig_ev.update_yaxes(type="log")
                st.plotly_chart(fig_ev, use_container_width=True)

