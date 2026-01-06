import math
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


# =========================
# Numerics
# =========================

def safe_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.float()
    p = torch.clamp(p, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)

def effective_support(p: torch.Tensor) -> torch.Tensor:
    return torch.exp(safe_entropy(p))

def topk_mass(p: torch.Tensor, k: int) -> torch.Tensor:
    p = p.float()
    k = min(k, p.shape[-1])
    vals, _ = torch.topk(p, k=k, dim=-1)
    return vals.sum(dim=-1)

def softmax_beta(scores: torch.Tensor, beta: float) -> torch.Tensor:
    # scores: [..., K]
    return torch.softmax((beta * scores).float(), dim=-1)

def var_under(p: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # p,x shape [K]
    p = p.float()
    x = x.float()
    m = (p * x).sum(dim=-1)
    m2 = (p * (x * x)).sum(dim=-1)
    return m2 - m * m


# =========================
# Score capture (robust)
# =========================

def _try_patch__attn_method(model) -> Dict[str, Any]:
    patched = 0
    for m in model.modules():
        if not hasattr(m, "_attn") or not callable(getattr(m, "_attn")):
            continue
        if getattr(m, "_gce_is_patched", False):
            continue

        orig = m._attn

        def make_new(attn_module, orig_fn):
            def new__attn(query, key, value, attention_mask=None, head_mask=None):
                scores = torch.matmul(query, key.transpose(-1, -2))  # [B,H,T,T]
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
            def new_forward(
                hidden_states,
                layer_past=None,
                attention_mask=None,
                head_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                use_cache=False,
                output_attentions=False,
            ):
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

                return orig_fwd(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

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


# =========================
# Precomputed cache loading (read-only)
# =========================

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
    attentions = []
    for i in range(L_attn):
        attentions.append(torch.tensor(z[f"attn_{i}"]))  # float16

    L_scores = int(z["L_scores"])
    scores = None
    if L_scores > 0:
        scores = []
        for i in range(L_scores):
            scores.append(torch.tensor(z[f"score_{i}"]))  # float16

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


# =========================
# Plotly helpers
# =========================

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

def fig_box(values_by_head: np.ndarray, title: str, y_title: str):
    # values_by_head: [H] or [H,L] -> flatten per head across L
    fig = go.Figure()
    if values_by_head.ndim == 1:
        fig.add_trace(go.Box(y=values_by_head, name="heads", boxpoints="all", jitter=0.3))
    else:
        # show one box per layer, values across heads
        H, L = values_by_head.shape
        for l in range(L):
            fig.add_trace(go.Box(y=values_by_head[:, l], name=f"ℓ={l}", boxpoints=False))
    fig.update_layout(title=title, yaxis_title=y_title, height=340)
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
        yaxis3=dict(title="Abs residual |r|", overlaying="y", side="right", showgrid=False, position=0.95),
        height=460,
        legend=dict(orientation="h"),
    )
    return fig


# =========================
# Core thermo stats (correct)
# =========================

@dataclass
class ThermoPerLayer:
    H: np.ndarray
    U: np.ndarray
    phi: np.ndarray
    omega: np.ndarray
    var_s: np.ndarray
    Neff: np.ndarray

def thermo_from_scores_row(scores_row: torch.Tensor, beta: float) -> ThermoPerLayer:
    """
    scores_row: [K] raw scores s_j
    Uses alpha_beta = softmax(beta*s).
    """
    s = scores_row.float()
    a = softmax_beta(s, beta=beta)
    H = float(safe_entropy(a).item())
    U = float((a * s).sum().item())
    phi = float(torch.logsumexp(beta * s, dim=-1).item())
    omega = float((-phi / beta))
    var_s = float(var_under(a, s).item())
    Neff = float(torch.exp(torch.tensor(H)).item())
    return H, U, phi, omega, var_s, Neff


def get_attention_row(attn: torch.Tensor, token_index: int, head: Optional[int]) -> torch.Tensor:
    # attn: [1,H,T,T]
    A = attn[0]  # [H,T,T]
    T = A.shape[-1]
    q = max(0, min(int(token_index), T - 1))
    row = A[:, q, :]  # [H,T]
    if head is None:
        return row.mean(dim=0)
    h = max(0, min(int(head), row.shape[0] - 1))
    return row[h]

def get_score_row(score: torch.Tensor, token_index: int, head: Optional[int]) -> torch.Tensor:
    # score: [1,H,T,T]
    S = score[0]
    T = S.shape[-1]
    q = max(0, min(int(token_index), T - 1))
    row = S[:, q, :]
    if head is None:
        return row.mean(dim=0)
    h = max(0, min(int(head), row.shape[0] - 1))
    return row[h]


# =========================
# Prompting: make distilgpt2 less deranged
# =========================

QA_FEWSHOT_PREFIX = """Q: Hey, how are you?
A: Pretty good—what’s up?

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

# compact icon submit button
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
    prompt_style = st.selectbox("Prompt style", ["Q/A few-shot (recommended for distilgpt2)", "Plain continuation"], index=0)
    max_new_tokens = st.slider("max_new_tokens", 1, 256, 64, 1)
    temperature = st.slider("temperature", 0.1, 2.0, 0.8, 0.1)
    top_p = st.slider("top_p", 0.1, 1.0, 0.95, 0.05)

    st.header("GCE parameters")
    beta = st.slider("β (used for αβ=softmax(βs) in thermo-consistent plots)", 0.1, 10.0, 1.0, 0.1)

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
    if hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

    model.to(device)
    model.eval()

    patch_info = install_score_capture(model)
    return tok, model, device, dtype, patch_info

tokenizer, model, device, dtype, patch_info = load_model_and_tokenizer(model_name, device_pref, dtype_pref)
st.sidebar.caption(f"Score capture: {patch_info}")


# =========================
# Session state init / seed load
# =========================

def reset_to_seed(seed: Dict[str, Any]) -> None:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if seed.get("prompt"):
        st.session_state.messages.append({"role": "user", "content": seed["prompt"]})
    if seed.get("assistant_text"):
        st.session_state.messages.append({"role": "assistant", "content": seed["assistant_text"]})
    st.session_state.diag = seed.get("diag", {})
    st.session_state.composer = ""
    st.session_state.has_seed = True

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "diag" not in st.session_state:
    st.session_state.diag = {}
if "composer" not in st.session_state:
    st.session_state.composer = ""
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
# Diagnostics compute for a sequence
# =========================

def compute_diagnostics_for_sequence(full_ids_1d: torch.Tensor, input_len: int) -> Dict[str, Any]:
    diag_input_ids = full_ids_1d.unsqueeze(0).to(device)
    attention_mask = torch.ones_like(diag_input_ids)

    # clear stale scores
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

    scores = collect_scores(model)
    scores = scores if (len(scores) >= len(attentions) and len(attentions) > 0) else None

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
# Render chat (messages) FIRST, input LAST
# =========================

st.markdown("### Chat")
for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# input at end
with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_input("Message", value="", placeholder="Type a message…", label_visibility="collapsed")
    submitted = st.form_submit_button("➤", type="primary")

if submitted:
    if not user_text.strip():
        st.warning("Type a message first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_text.strip()})
        with st.chat_message("user"):
            st.markdown(user_text.strip())

        with st.chat_message("assistant"):
            status = st.status("Generating…", expanded=True)
            try:
                status.write("Building prompt…")
                if prompt_style.startswith("Q/A"):
                    prompt_text = build_prompt_qa(st.session_state.messages)
                else:
                    prompt_text = build_prompt_plain(st.session_state.messages)

                status.write("Tokenizing…")
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
                st.markdown(assistant_text if assistant_text else "*[empty completion]*")

                st.session_state.messages.append({"role": "assistant", "content": assistant_text})
                st.session_state.diag = diag

            except Exception as e:
                status.update(label="Generation failed", state="error")
                st.exception(e)


# =========================
# Diagnostics section
# =========================

st.markdown("---")
st.subheader("Diagnostics")

with st.expander("❓ Global explainer: what is this measuring and when are identities valid?", expanded=False):
    st.markdown(
        r"""
We interpret an attention row as a Gibbs distribution over key positions \(j\).

### Two different \(\alpha\)s
- **Empirical attention** \(\alpha^{\text{model}}\): the actual weights the model used.
- **Thermo-consistent ensemble** \(\alpha_\beta := \mathrm{softmax}(\beta s)\), where \(s\) are the captured pre-softmax scores.

### Exact identity (Fenchel / log-sum-exp duality)
If \(\alpha_\beta = \mathrm{softmax}(\beta s)\), then:
\[
\phi(\beta) := \log\sum_j e^{\beta s_j} = \beta U(\beta) + H(\alpha_\beta),
\quad
U(\beta)=\langle s,\alpha_\beta\rangle.
\]
This **will not hold** if you plug in \(\alpha^{\text{model}}\) but compute \(s\) from a different object, or if you average heads before applying the identity.

### Head averaging caveat
The identity is nonlinear. You should compute \(H,U,\phi\) **per head**, then aggregate scalars across heads.

### β slider caveat
Changing β changes the ensemble. The identity uses \(\alpha_\beta=\mathrm{softmax}(\beta s)\). If you keep \(\alpha\) fixed while changing β, you’re no longer plotting a thermodynamic identity.
"""
    )


diag = st.session_state.diag
if not diag or not diag.get("attentions"):
    st.info("No diagnostics yet (run precompute.py or send a chat message).")
    st.stop()

attentions = diag["attentions"]
scores = diag.get("scores", None)
input_len = int(diag["input_len"])
new_ids = np.asarray(diag["new_ids"], dtype=np.int64)
steps = int(len(new_ids))

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


# ---------- Plot 1: Ensemble stats (from attention α)
with st.expander("ℹ️ Attention ensemble stats (H, N_eff, top-k mass)", expanded=False):
    st.markdown(
        r"""
Computed from the **empirical** attention weights \(\alpha^{\text{model}}\) for the selected token/step.

- \(H(\alpha)\): spread / uncertainty of attention.
- \(N_{\text{eff}}=\exp(H)\): effective number of attended positions.
- top1/top5 mass: how concentrated the attention is.
"""
    )

L = len(attentions)
layers = np.arange(L)

H_list, Neff_list, top1_list, top5_list = [], [], [], []
for l in range(L):
    p = get_attention_row(attentions[l], token_index_abs, chosen_head)
    H_list.append(float(safe_entropy(p).item()))
    Neff_list.append(float(torch.exp(torch.tensor(H_list[-1])).item()))
    top1_list.append(float(torch.max(p.float()).item()))
    top5_list.append(float(topk_mass(p, 5).item()))

fig_stats = fig_line(
    layers,
    {"H_att": np.array(H_list), "N_eff=exp(H)": np.array(Neff_list), "top1 mass": np.array(top1_list), "top5 mass": np.array(top5_list)},
    title=f"Attention ensemble stats vs layer (step t={step_idx})",
)
st.plotly_chart(fig_stats, use_container_width=True)


# ---------- Plot 2: Phase plot (U vs H)
with st.expander("ℹ️ Phase plot (U vs H): what does it mean?", expanded=False):
    st.markdown(
        r"""
Each point is a layer \(\ell\) mapped to \((H_\ell, U_\ell)\).

- \(H\): how diffuse attention is.
- \(U=\langle s,\alpha\rangle\): expected score under the attention distribution.

If you use **thermo-consistent** \(\alpha_\beta=\mathrm{softmax}(\beta s)\), this resembles a canonical equation-of-state curve:
higher β tends to reduce entropy and increase the “energy preference” for high-score keys.
"""
    )

if scores is None:
    st.warning("No pre-softmax scores captured, so U-based plots are unavailable. Fix score capture first (should now work with the updated patch).")
else:
    # For the phase plot we use either thermo-consistent alpha_beta (recommended) or empirical alpha
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


# ---------- Plot 3: Occupation spectrum
with st.expander("ℹ️ Occupation spectrum (sorted α): what to look for", expanded=False):
    st.markdown(
        r"""
This is the sorted attention distribution \(\alpha\) for a specific (layer, head).

A steep drop means a few keys dominate (localized phase); a flatter curve means diffuse attention (delocalized phase).
Log-y helps compare tails.
"""
    )

spec_layer = int(spectrum_layer)
spec_head = int(spectrum_head)
A = attentions[spec_layer][0]  # [H,T,T]
T = A.shape[-1]
q = max(0, min(int(token_index_abs), T - 1))
h = max(0, min(int(spec_head), A.shape[0] - 1))
p = A[h, q, :].float().cpu().numpy()
spec = np.sort(p)[::-1]
st.plotly_chart(fig_spectrum(spec, title=f"Occupation spectrum — step={step_idx}, layer={spec_layer}, head={spec_head}"), use_container_width=True)


# ---------- Plot 4: Decomposition + residuals (fixed)
with st.expander("ℹ️ Energy–entropy decomposition: why residuals appear", expanded=False):
    st.markdown(
        r"""
If \(\alpha_\beta = \mathrm{softmax}(\beta s)\), then exactly:
\[
\phi=\log\sum_j e^{\beta s_j} = \beta U + H.
\]

Residuals appear when you:
- mix \(\alpha^{\text{model}}\) with scores \(s\) that did not generate it,
- average heads before applying the identity,
- or your captured scores differ from the model’s true internal scores.

We plot both:
- signed residual \(r=\phi-(\beta U+H)\) (can be ±),
- absolute residual \(|r|\) (always ≥0).
"""
    )

if scores is not None:
    phi, betaU, Hdec, omega, signed_r, abs_r = [], [], [], [], [], []
    for l in range(L):
        srow = get_score_row(scores[l], token_index_abs, chosen_head)
        if decomp_mode.startswith("Thermo-consistent"):
            a = softmax_beta(srow, beta=float(beta))
        else:
            a = get_attention_row(attentions[l], token_index_abs, chosen_head).float()

        s = srow.float()
        ph = float(torch.logsumexp(float(beta) * s, dim=-1).item())
        U = float((a * s).sum().item())
        Hh = float(safe_entropy(a).item())
        om = float(-ph / float(beta))

        r = ph - (float(beta) * U + Hh)
        phi.append(ph)
        betaU.append(float(beta) * U)
        Hdec.append(Hh)
        omega.append(om)
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
else:
    st.info("Energy–entropy decomposition requires captured pre-softmax scores.")


# ---------- New: per-head thermo distributions
with st.expander("ℹ️ Per-head thermo diagnostics (distributions across heads)", expanded=False):
    st.markdown(
        r"""
We compute thermo-consistent scalars per head:
\(\alpha_\beta^{(h)}=\mathrm{softmax}(\beta s^{(h)})\),
then \(H^{(h)},U^{(h)},\phi^{(h)},\Omega^{(h)},\mathrm{Var}(s)^{(h)}\).

This avoids the “head averaging breaks nonlinear identities” issue.
"""
    )

if scores is not None:
    # use layer 0..L-1; for each layer we take all heads at selected token
    # We try to infer head count from tensor shape
    S0 = scores[0][0]  # [H,T,T]
    Hn = int(S0.shape[0])

    H_heads = np.zeros((Hn, L), dtype=float)
    U_heads = np.zeros((Hn, L), dtype=float)
    var_heads = np.zeros((Hn, L), dtype=float)

    for l in range(L):
        S = scores[l][0]  # [H,T,T]
        T = S.shape[-1]
        q = max(0, min(int(token_index_abs), T - 1))
        for h in range(Hn):
            srow = S[h, q, :].float()
            a = softmax_beta(srow, beta=float(beta))
            H_heads[h, l] = float(safe_entropy(a).item())
            U_heads[h, l] = float((a * srow).sum().item())
            var_heads[h, l] = float(var_under(a, srow).item())

    cA, cB, cC = st.columns(3)
    with cA:
        st.plotly_chart(fig_box(H_heads, f"H across heads (step={step_idx})", "H"), use_container_width=True)
    with cB:
        st.plotly_chart(fig_box(U_heads, f"U across heads (step={step_idx})", "U"), use_container_width=True)
    with cC:
        st.plotly_chart(fig_box(var_heads, f"Varα(s) across heads (heat capacity analogue)", "Var(s)"), use_container_width=True)
else:
    st.info("Per-head thermo diagnostics require captured pre-softmax scores.")


# ---------- 3D surfaces
if show_3d and steps > 0:
    with st.expander("ℹ️ Step×Layer surfaces: what they show", expanded=False):
        st.markdown(
            r"""
Surfaces show how a quantity evolves over:
- decoding step \(t\) (which generated token),
- layer \(\ell\).

These are the cleanest way to see “progressive sharpening” or “phase changes” across decoding.
"""
        )

    # entropy from empirical attention always available
    H_mat = np.zeros((steps, L), dtype=float)
    for t in range(steps):
        idx = int(input_len + t)
        for l in range(L):
            p = get_attention_row(attentions[l], idx, chosen_head)
            H_mat[t, l] = float(safe_entropy(p).item())

    s1, s2 = st.columns(2)
    with s1:
        st.plotly_chart(fig_surface(H_mat, "Entropy surface H(step, layer) [empirical α]", "Layer", "Generated step t", "H"), use_container_width=True)
    with s2:
        if scores is not None:
            U_mat = np.zeros((steps, L), dtype=float)
            for t in range(steps):
                idx = int(input_len + t)
                for l in range(L):
                    srow = get_score_row(scores[l], idx, chosen_head)
                    a = softmax_beta(srow, beta=float(beta))
                    U_mat[t, l] = float((a * srow.float()).sum().item())
            st.plotly_chart(fig_surface(U_mat, "U surface ⟨s,αβ⟩(step, layer) [thermo-consistent]", "Layer", "Generated step t", "U"), use_container_width=True)
        else:
            st.info("U surface requires pre-softmax scores.")


# ---------- Latent PCA + spectrum
if show_latent_pca:
    st.markdown("---")
    st.subheader("Post-LayerNorm PCA (3D) + Full spectrum")

    with st.expander("ℹ️ Post-LN PCA: what it means", expanded=False):
        st.markdown(
            r"""
We capture token representations **after each LayerNorm**, fit PCA(3) on all captured points (across tokens and LN sites),
then project. Because LayerNorm normalizes, points typically lie close to a sphere.

Right panel shows the **full eigenvalue spectrum** (via SVD on token covariance for the selected LN index).
"""
        )

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

