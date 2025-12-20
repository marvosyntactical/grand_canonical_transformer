import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import unicodedata

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Argos latent-geometry utilities (your file) ----
from argos_viz import (
    capture_layernorm_flow,
    pca3_fit,
    pca3_transform,
    to_unit_sphere,
    build_layer_slider_figure,
    color_gradient,
)

# -----------------------------
# Utilities
# -----------------------------

def to_cpu_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()

def safe_entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0)
    return -(p * torch.log(p)).sum(dim=-1)

def topk_mass(p: torch.Tensor, k: int) -> torch.Tensor:
    k = min(k, p.shape[-1])
    vals, _ = torch.topk(p, k=k, dim=-1)
    return vals.sum(dim=-1)

def effective_support_size(p: torch.Tensor) -> torch.Tensor:
    return torch.exp(safe_entropy(p))

# ------------------------- Text sanitization -------------------------
def remove_letters_with_diacritics(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("Ġ", " ").replace("▁", " ")
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        decomp = unicodedata.normalize("NFD", ch)
        if any(unicodedata.category(c) == "Mn" for c in decomp[1:]):
            i += 1
            while i < len(s) and unicodedata.category(unicodedata.normalize("NFD", s[i])) == "Mn":
                i += 1
            continue
        if i + 1 < len(s):
            nxt = unicodedata.normalize("NFD", s[i + 1])
            if any(unicodedata.category(c) == "Mn" for c in nxt):
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out).encode("ascii", "ignore").decode("utf-8")

def sanitize_tokens(tokens):
    return [remove_letters_with_diacritics(t) for t in tokens]

# -----------------------------
# Pre-softmax score capture (robust for GPT2/DistilGPT2 style)
# -----------------------------

def patch_gpt2_style_attn_scores(model) -> Dict[str, Any]:
    """
    Patch any module with a GPT2-style `_attn(query, key, value, attention_mask=None, head_mask=None)`
    to save the (masked, scaled) pre-softmax attention scores as `module._gce_scores`.
    """
    patched_modules = 0

    for m in model.modules():
        if not hasattr(m, "_attn") or not callable(getattr(m, "_attn", None)):
            continue
        if hasattr(m, "_gce_is_patched") and m._gce_is_patched:
            continue

        orig__attn = m._attn

        def make_new__attn(attn_module, orig_fn):
            def new__attn(query, key, value, attention_mask=None, head_mask=None):
                # query/key/value: [B,H,T,D]
                scores = torch.matmul(query, key.transpose(-1, -2))  # [B,H,Tq,Tk]

                # scaling if used
                if getattr(attn_module, "scale_attn_weights", False):
                    scores = scores / math.sqrt(value.size(-1))

                # causal mask (GPT2 stores a bool mask in `bias` [1,1,T,T])
                if hasattr(attn_module, "bias") and attn_module.bias is not None:
                    causal_mask = attn_module.bias[:, :, : scores.size(-2), : scores.size(-1)]
                    scores = torch.where(causal_mask, scores, torch.full_like(scores, -1e4))

                # additive attention mask
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
# Metrics computation
# -----------------------------

@dataclass
class PerLayerStats:
    H: np.ndarray          # [L]
    Neff: np.ndarray       # [L]
    top1: np.ndarray       # [L]
    top5: np.ndarray       # [L]
    U: Optional[np.ndarray]      # [L]
    phi: Optional[np.ndarray]    # [L]
    omega: Optional[np.ndarray]  # [L]

def compute_per_layer_stats_for_token(
    attentions: List[torch.Tensor],
    scores: Optional[List[torch.Tensor]],
    token_index: int,
    head_index: Optional[int],
    beta: float,
) -> PerLayerStats:
    """
    Compute stats vs layer for a single query token position (absolute index into sequence).
    attentions: list length L, each [B,H,T,T] post-softmax
    scores: list length >= L, each [B,H,T,T] pre-softmax (optional)
    """
    L = len(attentions)
    have_scores = scores is not None and len(scores) >= L

    H, Neff, top1, top5 = [], [], [], []
    U = [] if have_scores else None
    phi = [] if have_scores else None
    omega = [] if have_scores else None

    for l in range(L):
        A = attentions[l][0]  # [H,T,T]
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
        top1.append(float(torch.max(p).item()))
        top5.append(float(topk_mass(p, 5).item()))

        if have_scores:
            S = scores[l][0]  # [H,T,T]
            srow = S[:, qidx, :]
            if head_index is None:
                s = srow.mean(dim=0)
            else:
                s = srow[h_used]

            phi_val = torch.logsumexp(beta * s, dim=-1)
            U_val = (p * s).sum(dim=-1)  # <s, p>
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

def compute_sorted_spectrum(
    attentions: List[torch.Tensor],
    token_index: int,
    layer: int,
    head: int,
) -> Optional[np.ndarray]:
    if not attentions:
        return None
    L = len(attentions)
    if not (0 <= layer < L):
        return None

    A = attentions[layer][0]  # [H,T,T]
    T = A.shape[-1]
    qidx = max(0, min(int(token_index), T - 1))
    h = max(0, min(int(head), A.shape[0] - 1))
    p = A[h, qidx, :]
    return np.sort(to_cpu_np(p))[::-1]


# -----------------------------
# Plotly figures
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
    # Residual ε = |φ - (βU + H)|
    eps = np.abs(phi - (beta * U + H))

    x = np.arange(len(phi))
    fig = go.Figure()

    # Stacked components
    fig.add_trace(go.Bar(x=x, y=beta * U, name="β·U"))
    fig.add_trace(go.Bar(x=x, y=H, name="H"))

    # φ overlay
    fig.add_trace(go.Scatter(x=x, y=phi, mode="lines+markers", name="φ = log∑exp(βs)"))

    # ε on secondary axis
    fig.add_trace(go.Scatter(x=x, y=eps, mode="lines+markers", name="ε = |φ-(βU+H)|", yaxis="y2"))

    fig.update_layout(
        barmode="stack",
        title="Energy–Entropy Decomposition (φ vs βU + H) + Residual ε",
        xaxis_title="Layer",
        yaxis_title="Value",
        yaxis2=dict(
            title="Residual ε",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
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
    fig.update_layout(
        title=title,
        xaxis_title="Eigenvalue rank",
        yaxis_title="Eigenvalue",
        height=520,
        margin=dict(l=10, r=10, t=45, b=10),
    )
    fig.update_yaxes(type="log")
    return fig


# -----------------------------
# Prompting (fix distilgpt2 dialogue)
# -----------------------------

QA_FEWSHOT_PREFIX = """Q: Hey, how are you?
A: Pretty good—what’s up?

Q: Explain softmax in one sentence.
A: It turns scores into a probability distribution by exponentiating and normalizing.

"""

def build_prompt_plain(messages: List[Dict[str, str]]) -> str:
    # No role tags. Keep short, otherwise GPT2-family models start "format completion" games.
    parts = []
    for msg in messages[1:]:
        parts.append(msg["content"].strip())
    return "\n\n".join([p for p in parts if p]).strip() + "\n"

def build_prompt_qa(messages: List[Dict[str, str]]) -> str:
    # Convert history into Q/A lines but crucially: NOT "Assistant:" tokens.
    # distilgpt2 is happier continuing "A:" than "Assistant:".
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
# Streamlit app
# -----------------------------

st.set_page_config(page_title="The Transformer as Grand Canonical Ensemble.", layout="wide")
st.title("The Transformer as Grand Canonical Ensemble")
st.caption("Chat with a HuggingFace causal LM and visualize GCE-style diagnostics + post-LN latent geometry.")

# ---- Stylish icon send button ----
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

    # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    # Force attention weights to be materialized (otherwise out.attentions may contain None)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
    except TypeError:
        # Older transformers doesn't accept attn_implementation kwarg
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        if hasattr(model.config, "attn_implementation"):
            model.config.attn_implementation = "eager"

    model.to(device)
    model.eval()

    patch_info = install_score_capture(model)
    return tok, model, device, dtype, patch_info

tokenizer, model, device, dtype, patch_info = load_model_and_tokenizer(model_name, device_pref, dtype_pref)

st.sidebar.markdown("---")
st.sidebar.write("Score capture patch:", patch_info)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant."}]
if "diag" not in st.session_state:
    st.session_state.diag = {}

# Default prompt only on first interaction
if "has_sent_once" not in st.session_state:
    st.session_state.has_sent_once = False
if "composer" not in st.session_state:
    st.session_state.composer = "Hey, how are ya?"

# Render chat history
for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Composer
st.markdown("### Chat")
colA, colB = st.columns([6, 1])
with colA:
    st.session_state.composer = st.text_input("Message", value=st.session_state.composer, label_visibility="collapsed")
with colB:
    send = st.button("➤", type="primary", help="Send")

if send and st.session_state.composer.strip():
    user_prompt = st.session_state.composer.strip()

    # After the first send, leave composer empty
    st.session_state.has_sent_once = True
    st.session_state.composer = ""

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    told = user_prompt
    with st.chat_message("user"):
        st.markdown(told)

    if prompt_style.startswith("Q/A"):
        prompt_text = build_prompt_qa(st.session_state.messages)
    else:
        prompt_text = build_prompt_plain(st.session_state.messages)

    inputs = tokenizer(prompt_text, return_tensors="pt", padding=False).to(device)
    input_len = inputs["input_ids"].shape[-1]

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
    latest_assistant_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    if latest_assistant_text.startswith("A:"):
        latest_assistant_text = latest_assistant_text[2:].strip()

    st.session_state.messages.append({"role": "assistant", "content": latest_assistant_text})
    with st.chat_message("assistant"):
        st.markdown(latest_assistant_text if latest_assistant_text else "*[empty completion]*")

    # ---- Diagnostics forward pass on the final realized token sequence ----
    diag_input_ids = full_ids.unsqueeze(0) if full_ids.ndim == 1 else full_ids
    diag_input_ids = diag_input_ids.to(device)

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

    attentions = list(out.attentions) if out.attentions is not None else []
    scores = collect_patched_scores(model)
    scores = scores if (len(scores) >= len(attentions) and len(attentions) > 0) else None

    # ---- Latent PCA (post-LN) fit once on all LN activations ----
    pca_payload = {}
    if show_latent_pca:
        try:
            Z = capture_layernorm_flow(model, diag_input_ids, attention_mask=attention_mask)  # (T, Lnorm, D)
            Tlen, Lnorm, D = Z.shape
            Z_flat = Z.reshape(-1, D)  # (Tlen*Lnorm, D)

            pca_state = pca3_fit(Z_flat)
            X3 = pca3_transform(Z_flat, pca_state)  # (Tlen*Lnorm, 3)
            X3 = to_unit_sphere(X3).reshape(Tlen, Lnorm, 3)

            # ---- full eigenvalue spectra per LayerNorm (cov eigvals via SVD) ----
            eigvals_by_ln = []
            for ell in range(Lnorm):
                X = Z[:, ell, :]  # (T, D)
                X = X - X.mean(axis=0, keepdims=True)
                # X = U S V^T, cov eigenvals = S^2/(T-1)
                _, S_svd, _ = np.linalg.svd(X, full_matrices=False)
                denom = max(Tlen - 1, 1)
                eigvals = (S_svd ** 2) / denom
                eigvals_by_ln.append(eigvals.astype(np.float64))

            tok_text = tokenizer.convert_ids_to_tokens(diag_input_ids[0].tolist())
            tok_text = sanitize_tokens(tok_text)
            tok_colors = color_gradient(len(tok_text))

            fig_pca = build_layer_slider_figure(
                X3,
                token_text=tok_text,
                show_trajectories=bool(show_trajectories),
                token_colors=tok_colors,
                title="Post-LayerNorm token geometry (PCA3 on unit sphere)",
            )

            pca_payload = {
                "X3": X3,
                "tokens": tok_text,
                "colors": tok_colors,
                "Lnorm": int(Lnorm),
                "fig": fig_pca,
                "eigvals_by_ln": eigvals_by_ln,
            }
        except Exception as e:
            pca_payload = {"error": str(e)}

    st.session_state.diag = {
        "input_len": int(input_len),
        "new_ids": to_cpu_np(new_ids).astype(int),
        "attentions": attentions,
        "scores": scores,
        "diag_input_ids": diag_input_ids.detach().cpu(),
        "pca": pca_payload,
    }


# -----------------------------
# Diagnostics
# -----------------------------

st.markdown("---")
st.subheader("Diagnostics")

with st.expander("❓ What am I seeing? (GCE + Legendre view)", expanded=False):
    st.markdown(
        r"""
Treat a **single attention row** (one query token attending to all key positions) as a **grand-canonical ensemble** over discrete states \(j\) (the key positions).

**Pre-softmax scores / chemical potentials**
"""
    )
    st.latex(r"s_j = \langle q, k_j\rangle/\sqrt{d}")
    st.markdown("**Grand partition function** and **log-partition**:")
    st.latex(r"\Xi = \sum_j e^{\beta s_j}, \qquad \phi=\log \Xi = \log\sum_j e^{\beta s_j}")
    st.markdown("**Softmax** is the mean-occupancy map:")
    st.latex(r"\alpha_j = \frac{e^{\beta s_j}}{\sum_m e^{\beta s_m}} = \frac{\partial}{\partial(\beta s_j)}\log\sum_m e^{\beta s_m}")
    st.markdown(r"**Energy-like quantity \(U\)** (expected score under the ensemble):")
    st.latex(r"U = \langle s,\alpha\rangle = \sum_j \alpha_j\, s_j")
    st.markdown("**Grand potential proxy** (up to conventions):")
    st.latex(r"\Omega = -\frac{1}{\beta}\phi")
    st.markdown("**Entropy** of the attention ensemble:")
    st.latex(r"H(\alpha) = -\sum_j \alpha_j\log\alpha_j")
    st.markdown("**Energy–entropy decomposition** (identity):")
    st.latex(r"\phi = \beta U + H(\alpha)")
    st.markdown("### Across *successive decoding steps* (“multiple forward passes”)")
    st.markdown(
        r"""
At decoding step \(t\), the model forms chemical potentials \(s^{(t)}\) from the current latent state, computes the Gibbs distribution \(\alpha^{(t)}\), aggregates values into a new latent state, and finally produces **vocabulary** logits for the next token.

\[
s^{(t)} \;\xrightarrow{\text{Gibbs}}\; \alpha^{(t)} \;\xrightarrow{\mathbb{E}[V]}\; h^{(t+1)}.
\]

The 3D surfaces show how \(H, U, \Omega\) evolve over **(generated step × layer)**.
"""
    )

diag = st.session_state.diag
if not diag:
    st.info("Send a message to generate a response and compute diagnostics.")
else:
    attentions = diag["attentions"]
    scores = diag["scores"]
    input_len = diag["input_len"]
    new_ids = diag["new_ids"]
    steps = len(new_ids)

    if steps == 0:
        st.warning("No new tokens generated (steps=0). Increase max_new_tokens.")
    else:
        with st.sidebar:
            st.header("Forward-pass step")
            step_idx = st.slider("Generated token step t", 0, steps - 1, steps - 1, 1)
            tok_str = tokenizer.decode([int(new_ids[step_idx])], skip_special_tokens=False)
            st.caption(f"Selected token: `{tok_str}` (id={int(new_ids[step_idx])})")

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
                st.plotly_chart(
                    fig_phase(per_layer.U, per_layer.H, title=f"Phase plot (step t={step_idx}): U vs H"),
                    use_container_width=True,
                )
            else:
                st.warning("No pre-softmax scores captured; U/φ/Ω plots disabled for this run.")

        c3, c4 = st.columns(2)
        with c3:
            spec = compute_sorted_spectrum(
                attentions=attentions,
                token_index=token_index_abs,
                layer=int(spectrum_layer),
                head=int(spectrum_head),
            )
            if spec is not None:
                st.plotly_chart(
                    fig_spectrum(
                        spec,
                        title=f"Occupation spectrum (sorted α) — step={step_idx}, layer={int(spectrum_layer)}, head={int(spectrum_head)}",
                    ),
                    use_container_width=True,
                )
            else:
                st.info("Spectrum unavailable (check layer/head).")

        with c4:
            if per_layer.phi is not None and per_layer.U is not None:
                st.plotly_chart(
                    fig_waterfall(per_layer.phi, per_layer.U, per_layer.H, beta=float(beta)),
                    use_container_width=True,
                )
                eps = np.abs(per_layer.phi - (float(beta) * per_layer.U + per_layer.H))
                st.caption(f"Residual ε stats: mean={eps.mean():.3e}, max={eps.max():.3e}")
            else:
                st.info("Energy–entropy decomposition requires pre-softmax scores.")

        if show_3d:
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
                st.plotly_chart(
                    fig_surface(H_mat, "Entropy surface H(step, layer)", "Layer", "Generated step t", "H"),
                    use_container_width=True,
                )
            with s2:
                if U_mat is not None:
                    st.plotly_chart(
                        fig_surface(U_mat, "U surface ⟨s,α⟩(step, layer)", "Layer", "Generated step t", "U"),
                        use_container_width=True,
                    )
                else:
                    st.info("U surface requires pre-softmax scores.")
            if Om_mat is not None:
                st.plotly_chart(
                    fig_surface(Om_mat, "Grand potential proxy Ω(step, layer)", "Layer", "Generated step t", "Ω"),
                    use_container_width=True,
                )

        # ---- Latent PCA section ----
        if show_latent_pca:
            st.markdown("---")
            st.subheader("Post-LayerNorm PCA (3D) + Full spectrum")

            pca = diag.get("pca", {})
            if not pca:
                st.info("No PCA computed yet.")
            elif "error" in pca:
                st.warning(f"PCA capture failed: {pca['error']}")
            else:
                # Place plots first, slider at the bottom (controls both)
                plots_container = st.container()
                Lnorm = int(pca.get("Lnorm", 0))

                ln_sel = st.slider("LayerNorm index", 0, max(0, Lnorm - 1), 0, 1)

                with plots_container:
                    left, right = st.columns(2, gap="large")

                    with left:
                        fig = pca["fig"]
                        if fig.layout.sliders and len(fig.layout.sliders) > 0:
                            fig.layout.sliders[0].active = int(ln_sel)
                        st.plotly_chart(fig, use_container_width=True)

                    with right:
                        eigvals_by_ln = pca.get("eigvals_by_ln", None)
                        if eigvals_by_ln is None:
                            st.info("Eigenvalue spectrum unavailable.")
                        else:
                            ev = eigvals_by_ln[int(ln_sel)]
                            st.plotly_chart(
                                fig_eigspectrum(ev, title=f"Eigenvalue spectrum (LayerNorm {ln_sel})"),
                                use_container_width=True,
                            )

st.markdown("---")
st.caption(
    "Interpretation: each decoding step forms chemical potentials s → Gibbs α via softmax → expected value update → next latent → next token logits."
)

