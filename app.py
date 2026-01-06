import math
import os
import json
from dataclasses import dataclass
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


# -----------------------------
# Numerics
# -----------------------------

def safe_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Critical for cached float16 attentions: eps underflows to 0 otherwise -> NaNs
    p = p.float()
    p = torch.clamp(p, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)

def topk_mass(p: torch.Tensor, k: int) -> torch.Tensor:
    p = p.float()
    k = min(k, p.shape[-1])
    vals, _ = torch.topk(p, k=k, dim=-1)
    return vals.sum(dim=-1)

def effective_support_size(p: torch.Tensor) -> torch.Tensor:
    return torch.exp(safe_entropy(p))


# -----------------------------
# Pre-softmax score capture (GPT2/DistilGPT2-style)
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
# Precomputed cache loading (read-only)
# -----------------------------

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
        a = z[f"attn_{i}"]  # float16
        attentions.append(torch.tensor(a))  # [1,H,T,T]

    L_scores = int(z["L_scores"])
    scores = None
    if L_scores > 0:
        scores = []
        for i in range(L_scores):
            s = z[f"score_{i}"]
            scores.append(torch.tensor(s))

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


# -----------------------------
# Stats
# -----------------------------

@dataclass
class PerLayerStats:
    H: np.ndarray
    Neff: np.ndarray
    top1: np.ndarray
    top5: np.ndarray
    U: Optional[np.ndarray]
    phi: Optional[np.ndarray]
    omega: Optional[np.ndarray]

def compute_per_layer_stats_for_token(
    attentions: List[torch.Tensor],      # list of [1,H,T,T]
    scores: Optional[List[torch.Tensor]],
    token_index: int,
    head_index: Optional[int],
    beta: float,
) -> PerLayerStats:
    L = len(attentions)
    have_scores = scores is not None and len(scores) >= L

    H, Neff, top1, top5 = [], [], [], []
    U = [] if have_scores else None
    phi = [] if have_scores else None
    omega = [] if have_scores else None

    for l in range(L):
        A = attentions[l]
        if A is None:
            continue
        A = A[0]  # [H,T,T]
        T = A.shape[-1]
        qidx = max(0, min(int(token_index), T - 1))

        row = A[:, qidx, :]  # [H,T]
        if head_index is None:
            p = row.mean(dim=0)
            h_used = 0
        else:
            h_used = max(0, min(int(head_index), row.shape[0] - 1))
            p = row[h_used]

        H.append(float(safe_entropy(p).item()))
        Neff.append(float(effective_support_size(p).item()))
        top1.append(float(torch.max(p.float()).item()))
        top5.append(float(topk_mass(p, 5).item()))

        if have_scores:
            S = scores[l][0]  # [H,T,T]
            srow = S[:, qidx, :]
            if head_index is None:
                s = srow.mean(dim=0)
            else:
                s = srow[h_used]

            # compute with float32 for stability
            s = s.float()
            p_f = p.float()

            phi_val = torch.logsumexp(beta * s, dim=-1)
            U_val = (p_f * s).sum(dim=-1)
            omega_val = -phi_val / beta

            U.append(float(U_val.item()))
            phi.append(float(phi_val.item()))
            omega.append(float(omega_val.item()))

    return PerLayerStats(
        H=np.array(H),
        Neff=np.array(Neff),
        top1=np.array(top1),
        top5=np.array(top5),
        U=np.array(U) if U is not None else None,
        phi=np.array(phi) if phi is not None else None,
        omega=np.array(omega) if omega is not None else None,
    )

def compute_sorted_spectrum(attentions: List[torch.Tensor], token_index: int, layer: int, head: int) -> Optional[np.ndarray]:
    if not attentions:
        return None
    if not (0 <= layer < len(attentions)):
        return None
    A = attentions[layer]
    if A is None:
        return None
    A = A[0]  # [H,T,T]
    T = A.shape[-1]
    qidx = max(0, min(int(token_index), T - 1))
    h = max(0, min(int(head), A.shape[0] - 1))
    p = A[h, qidx, :].float().cpu().numpy()
    return np.sort(p)[::-1]


# -----------------------------
# Plotly
# -----------------------------

def fig_line(x, ys: Dict[str, np.ndarray], title: str, x_title="Layer", y_title="Value"):
    fig = go.Figure()
    for name, y in ys.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=name))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, height=340)
    return fig

def fig_phase(U: np.ndarray, H: np.ndarray, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=H, y=U, mode="lines+markers", name="layers"))
    fig.update_layout(title=title, xaxis_title="Entropy H", yaxis_title="U = ⟨s,α⟩", height=340)
    return fig

def fig_spectrum(spec: np.ndarray, title: str):
    x = np.arange(1, len(spec) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=spec, mode="lines", name="sorted α"))
    fig.update_layout(title=title, xaxis_title="Rank", yaxis_title="Attention weight", height=340)
    fig.update_yaxes(type="log")
    return fig

def fig_waterfall(phi: np.ndarray, U: np.ndarray, H: np.ndarray, beta: float):
    eps = np.abs(phi - (beta * U + H))
    x = np.arange(len(phi))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=beta * U, name="β·U"))
    fig.add_trace(go.Bar(x=x, y=H, name="H"))
    fig.add_trace(go.Scatter(x=x, y=phi, mode="lines+markers", name="φ = log∑exp(βs)"))
    fig.add_trace(go.Scatter(x=x, y=eps, mode="lines+markers", name="ε = |φ-(βU+H)|", yaxis="y2"))
    fig.update_layout(
        barmode="stack",
        title="Energy–Entropy Decomposition (φ vs βU + H) + Residual ε",
        xaxis_title="Layer",
        yaxis_title="Value",
        yaxis2=dict(title="Residual ε", overlaying="y", side="right", showgrid=False),
        height=420,
        legend=dict(orientation="h"),
    )
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

def fig_eigspectrum(eigvals: np.ndarray, title: str):
    ev = np.array(eigvals, dtype=float)
    ev = np.sort(ev)[::-1]
    x = np.arange(1, len(ev) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=ev, mode="lines", name="eigenvalues"))
    fig.update_layout(title=title, xaxis_title="Eigenvalue rank", yaxis_title="Eigenvalue", height=520)
    fig.update_yaxes(type="log")
    return fig


# -----------------------------
# Prompting
# -----------------------------

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


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="The Transformer as Grand Canonical Ensemble", layout="wide")
st.title("The Transformer as Grand Canonical Ensemble")
st.caption("Chat with a HuggingFace causal LM and visualize GCE-style diagnostics + post-LN latent geometry.")

# Icon send button styling
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
    prompt_style = st.selectbox("Prompt style", ["Plain continuation", "Q/A few-shot (recommended for distilgpt2)"], index=0)

    st.header("GCE parameters")
    beta = st.slider("β (inverse temperature used for φ/logsumexp)", 0.1, 10.0, 1.0, 0.1)
    max_new_tokens = st.slider("max_new_tokens", 1, 256, 64, 1)
    temperature = st.slider("generation temperature", 0.1, 2.0, 0.8, 0.1)
    top_p = st.slider("top_p", 0.1, 1.0, 0.95, 0.05)

    st.header("Visualization")
    head_mode = st.selectbox("Head selection", ["mean over heads", "specific head"], index=0)
    head_index = st.number_input("Head index", value=0, step=1)
    spectrum_layer = st.number_input("Spectrum layer", value=0, step=1)
    spectrum_head = st.number_input("Spectrum head", value=0, step=1)
    show_3d = st.checkbox("Show 3D surfaces (step × layer)", value=True)

    st.header("Latent PCA")
    show_latent_pca = st.checkbox("Show post-LN PCA (3D)", value=True)
    show_trajectories = st.checkbox("Show token trajectories", value=True)

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

    # Force eager attention so attentions exist (and hooks are hit)
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

# -----------------------------
# Session state init
# -----------------------------

def reset_to_seed(seed: Dict[str, Any]) -> None:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
    if seed.get("prompt") and seed.get("assistant_text"):
        st.session_state.messages.append({"role": "user", "content": seed["prompt"]})
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

# Load seed exactly once per session (latest precomputed)
if not st.session_state.has_seed:
    latest = find_latest_cache_for_model(model_name)
    if latest is not None:
        cached = load_npz_cache(latest)
        meta = cached.get("meta", {}) or {}
        prompt = meta.get("prompt", "Hey, how are ya?")
        assistant_text = meta.get("assistant_text", "")

        # Build PCA fig from cached X3 if present
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
        reset_to_seed(st.session_state.seed)
    else:
        # No cache found; seed with a single user line (no assistant), but chat still works.
        st.session_state.seed = {"prompt": "Hey, how are ya?", "assistant_text": "", "diag": {}}
        reset_to_seed(st.session_state.seed)

# Clear button resets to seed without touching precomputed
if clear:
    reset_to_seed(st.session_state.seed)
    st.rerun()


# -----------------------------
# Render chat history
# -----------------------------

st.markdown("### Chat")
for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Chat input (use a FORM so input+submit is atomic and doesn't go "empty" on rerun)
with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_input("Message", value="", placeholder="Type a message…", label_visibility="collapsed")
    send_cols = st.columns([6, 1])
    with send_cols[1]:
        submitted = st.form_submit_button("➤", type="primary")

# If you want the “default prompt only initially” behavior:
# show it in the seed message, not in the input box.
# (Input box stays empty by default.)

# -----------------------------
# Run generation + diagnostics
# -----------------------------

def compute_diagnostics_for_sequence(full_ids_1d: torch.Tensor, input_len: int) -> Dict[str, Any]:
    diag_input_ids = full_ids_1d.unsqueeze(0).to(device)
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

if submitted:
    if not user_text.strip():
        st.warning("Type a message first.")
    else:
        # Append user message immediately
        st.session_state.messages.append({"role": "user", "content": user_text.strip()})
        with st.chat_message("user"):
            st.markdown(user_text.strip())

        # Progress UI
        prog = st.progress(0, text="Generating…")
        try:
            prog.progress(10, text="Building prompt…")
            if prompt_style.startswith("Q/A"):
                prompt_text = build_prompt_qa(st.session_state.messages)
            else:
                prompt_text = build_prompt_plain(st.session_state.messages)

            prog.progress(25, text="Tokenizing…")
            inputs = tokenizer(prompt_text, return_tensors="pt", padding=False).to(device)
            input_len = int(inputs["input_ids"].shape[-1])

            prog.progress(45, text="Sampling completion…")
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

            full_ids = gen.sequences[0]  # [T]
            new_ids = full_ids[input_len:]
            assistant_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            if assistant_text.startswith("A:"):
                assistant_text = assistant_text[2:].strip()

            prog.progress(70, text="Computing diagnostics…")
            diag = compute_diagnostics_for_sequence(full_ids, input_len)

            prog.progress(95, text="Updating UI…")
            st.session_state.messages.append({"role": "assistant", "content": assistant_text})
            st.session_state.diag = diag

            with st.chat_message("assistant"):
                st.markdown(assistant_text if assistant_text else "*[empty completion]*")

            prog.progress(100, text="Done.")
        except Exception as e:
            prog.empty()
            st.error("Generation failed.")
            st.exception(e)
        finally:
            try:
                prog.empty()
            except Exception:
                pass


# -----------------------------
# Diagnostics section
# -----------------------------

st.markdown("---")
st.subheader("Diagnostics")

with st.expander("❓ What am I seeing? (GCE + Legendre view)", expanded=False):
    st.markdown(
        r"""
Treat a **single attention row** (one query token attending to all key positions) as a **grand-canonical ensemble** over discrete states \(j\) (the key positions).
"""
    )
    st.latex(r"s_j = \langle q, k_j\rangle/\sqrt{d}")
    st.latex(r"\Xi = \sum_j e^{\beta s_j}, \qquad \phi=\log\Xi=\log\sum_j e^{\beta s_j}")
    st.latex(r"\alpha_j = \frac{e^{\beta s_j}}{\sum_m e^{\beta s_m}}")
    st.latex(r"U = \langle s,\alpha\rangle = \sum_j \alpha_j s_j")
    st.latex(r"\Omega = -\frac{1}{\beta}\phi")
    st.latex(r"H(\alpha) = -\sum_j \alpha_j\log\alpha_j")
    st.latex(r"\phi = \beta U + H(\alpha)")

diag = st.session_state.diag
if not diag or not diag.get("attentions"):
    st.info("No diagnostics yet (run precompute.py or send a chat message).")
else:
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

    per_layer = compute_per_layer_stats_for_token(
        attentions=attentions,
        scores=scores,
        token_index=token_index_abs,
        head_index=chosen_head,
        beta=float(beta),
    )
    L = len(per_layer.H)
    layers = np.arange(L)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            fig_line(
                layers,
                {
                    "H_att": per_layer.H,
                    "N_eff=exp(H)": per_layer.Neff,
                    "top1 mass": per_layer.top1,
                    "top5 mass": per_layer.top5,
                },
                title=f"Attention ensemble stats vs layer (step t={step_idx})",
            ),
            use_container_width=True,
        )
    with c2:
        if per_layer.U is not None:
            st.plotly_chart(fig_phase(per_layer.U, per_layer.H, title=f"Phase plot (step t={step_idx}): U vs H"), use_container_width=True)
        else:
            st.warning("No pre-softmax scores captured; U/φ/Ω plots disabled for this run.")

    c3, c4 = st.columns(2)
    with c3:
        spec = compute_sorted_spectrum(attentions, token_index_abs, int(spectrum_layer), int(spectrum_head))
        if spec is not None:
            st.plotly_chart(fig_spectrum(spec, title=f"Occupation spectrum — step={step_idx}, layer={int(spectrum_layer)}, head={int(spectrum_head)}"), use_container_width=True)
        else:
            st.info("Spectrum unavailable (check layer/head).")

    with c4:
        if per_layer.phi is not None and per_layer.U is not None:
            st.plotly_chart(fig_waterfall(per_layer.phi, per_layer.U, per_layer.H, beta=float(beta)), use_container_width=True)
            eps = np.abs(per_layer.phi - (float(beta) * per_layer.U + per_layer.H))
            st.caption(f"Residual ε stats: mean={eps.mean():.3e}, max={eps.max():.3e}")
        else:
            st.info("Energy–entropy decomposition requires pre-softmax scores.")

    if show_3d and steps > 0:
        H_mat = np.zeros((steps, L), dtype=float)
        U_mat = np.zeros((steps, L), dtype=float) if scores is not None else None
        Om_mat = np.zeros((steps, L), dtype=float) if scores is not None else None

        for t in range(steps):
            idx = int(input_len + t)
            pl = compute_per_layer_stats_for_token(attentions, scores, idx, chosen_head, float(beta))
            H_mat[t, :] = pl.H
            if scores is not None and pl.U is not None and pl.omega is not None:
                U_mat[t, :] = pl.U
                Om_mat[t, :] = pl.omega

        st.markdown("### Step × Layer surfaces")
        s1, s2 = st.columns(2)
        with s1:
            st.plotly_chart(fig_surface(H_mat, "Entropy surface H(step, layer)", "Layer", "Generated step t", "H"), use_container_width=True)
        with s2:
            if U_mat is not None:
                st.plotly_chart(fig_surface(U_mat, "U surface ⟨s,α⟩(step, layer)", "Layer", "Generated step t", "U"), use_container_width=True)
            else:
                st.info("U surface requires pre-softmax scores.")
        if Om_mat is not None:
            st.plotly_chart(fig_surface(Om_mat, "Grand potential proxy Ω(step, layer)", "Layer", "Generated step t", "Ω"), use_container_width=True)

    if show_latent_pca:
        st.markdown("---")
        st.subheader("Post-LayerNorm PCA (3D) + Full spectrum")
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
                    st.plotly_chart(fig_eigspectrum(ev, title=f"Eigenvalue spectrum (LayerNorm {ln_sel})"), use_container_width=True)

