
"""
argos_viz.py — clean, minimal latent-trajectory visualisation utilities
for decoder-only Transformers (e.g., GPT-2 family) using PyTorch + Plotly.

- Captures LayerNorm (or RMSNorm) activations at each block (pre/post attn/MLP)
- Projects to 3D with a tiny SVD-based PCA (no sklearn dependency)
- Builds Plotly figures with a layer slider; optionally shows token trajectories
- Designed to be called repeatedly step-by-step during generation

Notes
-----
• We compute a fresh PCA at every step by default. This makes earlier token
  points shift slightly as the basis changes — matching the "evolves anew"
  feel. You can freeze the PCA basis by passing a (mean, comps) tuple.
• Works best with small models (distilgpt2 by default). For other architectures
  we try to hook nn.LayerNorm or RMSNorm modules under each block.

Author: you + ChatGPT
"""

from __future__ import annotations

import math
import typing as T
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go

import colorsys

import unicodedata

# ------------------------- Text sanitization -------------------------

def remove_letters_with_diacritics(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("Ġ", " ").replace("▁", " ")
    out = []
    i = 0
    while i < len(s):
        ch = s[i]
        if len(ch) == 1:  # Ensure it's a single character
            try:
                decomp = unicodedata.normalize("NFD", ch)
                # Skip characters with diacritics (category 'Mn' is for marks)
                if any(unicodedata.category(c) == "Mn" for c in decomp[1:]):
                    i += 1
                    continue
            except (TypeError, ValueError):
                # Skip any invalid characters and move to the next one
                i += 1
                continue
        if i + 1 < len(s):
            nxt = unicodedata.normalize("NFD", s[i + 1])
            if any(unicodedata.category(c) == "Mn" for c in nxt):
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out).encode("ascii", "ignore").decode("ascii")

def sanitize_tokens(tokens):
    return [remove_letters_with_diacritics(t) for t in tokens]

def sanitize_figure_text(fig: go.Figure):
    def fix(tr):
        if hasattr(tr, "text") and tr.text is not None:
            if isinstance(tr.text, (list, tuple)):
                tr.text = [remove_letters_with_diacritics(x) for x in tr.text]
            else:
                tr.text = remove_letters_with_diacritics(tr.text)
    for tr in fig.data:
        fix(tr)
    if fig.frames:
        for fr in fig.frames:
            for tr in fr.data:
                fix(tr)
    return fig


def color_gradient(n: int, span: int = 1024):
    """
    Smooth HSV gradient across 'span' positions (default 1024).
    Tokens 0..n-1 map to evenly spaced hues in [0, span).
    """
    out = []
    for i in range(n):
        h = (i % span) / float(span)          # 0..1
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.98)  # pleasant saturation/value
        out.append("#%02x%02x%02x" % (int(r*255), int(g*255), int(b*255)))
    return out


# ----------------------------- PCA utilities -----------------------------

@dataclass
class PCA3:
    mean: np.ndarray  # (d,)
    comps: np.ndarray # (d, 3)

def pca3_fit(X: np.ndarray) -> PCA3:
    """
    Fit a 3D PCA using numpy SVD (no sklearn).
    X: (N, d)
    returns PCA3(mean, comps) where comps has shape (d, 3)
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0, keepdims=True)  # (1, d)
    Xc = X - mu
    # economy SVD on (N, d): use covariance via SVD on Xc
    # We just need top-3 right singular vectors
    # For stability on tall matrices, compute SVD of Xc using full_matrices=False
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:3].T  # (d, 3)
    return PCA3(mean=mu.squeeze(0), comps=comps)

def pca3_transform(X: np.ndarray, p: PCA3) -> np.ndarray:
    """
    Project to 3D using a fitted PCA3.
    Returns (N, 3).
    """
    Xc = X - p.mean[None, :]
    Z = Xc @ p.comps
    return Z

def to_unit_sphere(X3: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(X3, axis=1, keepdims=True) + eps
    return X3 / n


# ---------------------- LayerNorm capture / tracing ----------------------

def _iter_norm_modules(model: nn.Module):
    """Collect normalization modules under transformer blocks in forward order, deduped."""
    blocks = None
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        blocks = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "layers"):
        blocks = model.layers

    seen = set()

    def yield_once(m):
        mid = id(m)
        if mid not in seen:
            seen.add(mid)
            yield m

    if blocks is None:
        for m in model.modules():
            if isinstance(m, nn.LayerNorm) or m.__class__.__name__.lower().endswith("rmsnorm"):
                yield from yield_once(m)
        return

    for b in blocks:
        # canonical names first
        for name in ("ln_1", "ln_2", "input_layernorm", "post_attention_layernorm"):
            if hasattr(b, name):
                yield from yield_once(getattr(b, name))
        # fallbacks inside the block (but deduped)
        for _, m in b.named_modules():
            if isinstance(m, nn.LayerNorm) or m.__class__.__name__.lower().endswith("rmsnorm"):
                yield from yield_once(m)


@torch.no_grad()
def capture_layernorm_flow(model: nn.Module, input_ids: torch.Tensor, attention_mask: T.Optional[torch.Tensor]=None) -> np.ndarray:
    """
    Run one full forward pass and capture the activation *after* each norm module.
    Returns:
      Z: (T, L, D)  for a single sequence (batch size must be 1)
    """
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1, "Use batch size = 1 for tracing"
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    acts: T.List[torch.Tensor] = []
    handles = []

    def hook(_mod, _inp, out):
        # out shape: (B, T, D)
        acts.append(out.detach().float().cpu())

    # register hooks in canonical order
    for m in _iter_norm_modules(model):
        handles.append(m.register_forward_hook(hook))

    # Forward (no cache so shapes stay (B,T,D) everywhere)
    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    for h in handles:
        h.remove()

    # acts: list of (1, T_i, D). Make T consistent.
    if len(acts) == 0:
        raise RuntimeError("No LayerNorm/RMSNorm activations were captured. Unsupported model architecture?")

    T_min = min(a.shape[1] for a in acts)
    D = acts[0].shape[2]
    acts = [a[:, -T_min:, :].contiguous() for a in acts]  # right-align

    # (L, 1, T, D) -> (T, L, D)
    Z = torch.stack(acts, dim=0).squeeze(1).permute(1, 0, 2).contiguous()
    return Z.numpy()

# -------------------------- Plotly figure maker --------------------------

def color_palette(n: int) -> T.List[str]:
    # Pleasant categorical palette (tab10 extended)
    base = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
    ]
    if n <= len(base):
        return base[:n]
    # extend by lightening
    out = []
    for i in range(n):
        c = base[i % len(base)]
        out.append(c)
    return out

def build_layer_slider_figure(
    X: np.ndarray,              # (T, L, 3) projected points on unit sphere
    token_text: T.List[str],    # length T labels
    show_trajectories: bool=True,
    token_colors: T.Optional[T.List[str]]=None,
    title: str="Latent flow across layer norms"
) -> go.Figure:
    """
    Build a Plotly figure with one frame per layer. Each frame shows all tokens'
    positions at that layer. Optionally overlays trajectories (lines) from layer 0 to L.
    """
    Tlen, L, _ = X.shape
    if token_colors is None:
        token_colors = color_palette(Tlen)

    # Build per-token color array for markers
    marker_colors = np.array(token_colors)

    # Precompute line segments for trajectories: (x, y, z) arrays with None separators
    def traj_until(layer_idx: int):
        xs, ys, zs = [], [], []
        for t in range(Tlen):
            xs.extend(X[t, :layer_idx+1, 0].tolist()); xs.append(None)
            ys.extend(X[t, :layer_idx+1, 1].tolist()); ys.append(None)
            zs.extend(X[t, :layer_idx+1, 2].tolist()); zs.append(None)
        return xs, ys, zs

    frames = []
    for l in range(L):
        pts = X[:, l, :]  # (T, 3)
        data = []

        # marker scatter
        data.append(go.Scatter3d(
            x=pts[:,0], y=pts[:,1], z=pts[:,2],
            mode="markers+text",
            text=token_text,
            textposition="top center",
            marker=dict(size=4, opacity=0.95, color=marker_colors),
            hovertemplate="token: %{text}<extra></extra>",
            name="tokens",
            showlegend=False,
        ))

        if show_trajectories and l > 0:
            xs, ys, zs = traj_until(l)
            data.append(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(width=1),
                opacity=0.35,
                name="trajectories",
                showlegend=False,
            ))

        frames.append(go.Frame(data=data, name=f"LN {l}"))

    layout = go.Layout(
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
        width=820, height=700, margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[f"LN {l}"], dict(mode="immediate", frame=dict(duration=0, redraw=True), transition=dict(duration=0))],
                        label=f"LN {l}") for l in range(L)],
            active=0, x=0, y=0, len=1.0
        )],
        updatemenus=[dict(type="buttons", showactive=False, buttons=[
            dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=450, redraw=True), transition=dict(duration=0))])
        ])]
    )

    fig = go.Figure(data=frames[0].data, frames=frames, layout=layout)
    return sanitize_figure_text(fig)


# ---------------------- High-level run for one step ----------------------

@dataclass
class StepViz:
    tokens: T.List[str]            # length T
    colors: T.List[str]            # length T hex colors
    X: np.ndarray                  # (T, L, 3) sphere coords
    figure_json: dict              # Plotly figure as JSON-compatible dict

def project_sequence_to_sphere(
    model: nn.Module,
    tokenizer,
    input_ids: torch.Tensor,              # (1, T)
    pca_state: T.Optional[PCA3]=None,
    show_trajectories: bool=True,
    re_fit_pca_each_step: bool=True
) -> T.Tuple[StepViz, PCA3]:
    """
    Capture LN activations for a full sequence and project to unit sphere.
    If pca_state is provided and re_fit_pca_each_step=False, we use it to keep
    the basis fixed across steps.
    """
    # 1) capture
    Z = capture_layernorm_flow(model, input_ids)  # (T, L, D)
    Tlen, L, D = Z.shape

    # 2) PCA basis
    # --- PCA basis on ALL layernorm activations ---
    Z_flat = Z.reshape(-1, Z.shape[-1])  # (T*L, D)
    if (pca_state is None) or re_fit_pca_each_step:
        pca_state = pca3_fit(Z_flat)
    X3 = pca3_transform(Z_flat, pca_state)  # (T*L, 3)
    X3 = to_unit_sphere(X3).reshape(Tlen, L, 3)

    # --- Token labels + smooth gradient colours ---
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    colors = color_gradient(len(tokens))  # << gradient instead of categorical

    # --- Figure ---
    fig = build_layer_slider_figure(X3, tokens, show_trajectories=show_trajectories, token_colors=colors)
    return StepViz(tokens=tokens, colors=colors, X=X3, figure_json=fig.to_plotly_json()), pca_state
# ----------------------- Token-by-token generation -----------------------

@torch.no_grad()
def generate_steps(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
    pca_freeze_after_first: bool = False,
    show_trajectories: bool = True,
) -> T.Tuple[T.List[StepViz], str]:
    """
    Auto-regressively generate tokens step-by-step, returning a StepViz for each
    intermediate sequence (prompt + k new tokens).

    Returns (steps, full_decoded_text).
    """
    device = next(model.parameters()).device
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    full_ids = input_ids.clone()
    steps: T.List[StepViz] = []

    pca_state: T.Optional[PCA3] = None
    for k in range(max_new_tokens):
        # Project current sequence
        step_viz, pca_state = project_sequence_to_sphere(
            model, tokenizer, full_ids, pca_state=pca_state,
            show_trajectories=show_trajectories,
            re_fit_pca_each_step=not pca_freeze_after_first or (pca_state is None)
        )
        steps.append(step_viz)

        # One decoding step
        out = model(input_ids=full_ids, use_cache=True)
        logits = out.logits[:, -1, :] / temperature
        next_id = torch.softmax(logits, dim=-1).multinomial(num_samples=1)
        full_ids = torch.cat([full_ids, next_id], dim=1)

    # Final decode
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    return steps, text
