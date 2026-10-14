"""Stage I causal-feasibility utilities: residual-stream forward hooks + logit lens.

Decoder-layer indexing (Qwen3-0.6B): base_model.model.layers[i], i in [0, 27].
output_hidden_states returns 29 tensors; hidden_states[k] is the OUTPUT of layers[k-1]
(hidden_states[0] = embeddings). The helpers here take the DECODER-LAYER index i
directly; a hook on layer i modifies the residual that becomes hidden_states[i+1].

These are also the write-path primitives Stage II reuses (add a direction / project a
direction out, optionally only at selected positions).
"""

from __future__ import annotations

import contextlib
import torch


def decoder_layers(model):
    return _core(model).layers


def _core(model):
    """The decoder stack container (.layers, .norm). Standard across Llama/Gemma/Mistral/
    Phi/Qwen/OLMo (base_model.model); fall back to base_model if a family nests differently."""
    bm = model.base_model
    return bm.model if hasattr(bm, "model") and hasattr(bm.model, "layers") else bm


def final_norm(model):
    return _core(model).norm


def lm_head(model):
    bm = model.base_model
    return bm.lm_head if hasattr(bm, "lm_head") else bm.get_output_embeddings()


def _split(output):
    """Qwen3DecoderLayer may return a tensor or a tuple (hidden, ...)."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


@contextlib.contextmanager
def residual_hooks(model, layer_fns: dict):
    """Temporarily install forward hooks. ``layer_fns`` maps a decoder-layer index to
    a function fn(hidden:(b,s,d))->(b,s,d) that rewrites that layer's output residual."""
    handles = []
    layers = decoder_layers(model)

    def make_hook(f):
        def hook(module, inp, output):
            h, rest = _split(output)
            h2 = f(h)
            return h2 if rest is None else (h2,) + rest
        return hook

    try:
        for idx, fn in layer_fns.items():
            handles.append(layers[idx].register_forward_hook(make_hook(fn)))
        yield
    finally:
        for h in handles:
            h.remove()


def add_direction(vec: torch.Tensor, coeff: float = 1.0, positions=None):
    """fn that adds coeff*vec to the residual (at `positions`, or all positions)."""
    def fn(h):
        v = vec.to(h.dtype).to(h.device)
        if positions is None:
            return h + coeff * v
        h = h.clone()
        h[:, positions, :] = h[:, positions, :] + coeff * v
        return h
    return fn


def project_out(direction: torch.Tensor, positions=None):
    """fn that removes the component along `direction` (ablation), at `positions` or all."""
    d = direction / (direction.norm() + 1e-8)

    def fn(h):
        dd = d.to(h.dtype).to(h.device)
        if positions is None:
            coeff = (h @ dd).unsqueeze(-1)
            return h - coeff * dd
        h = h.clone()
        seg = h[:, positions, :]
        coeff = (seg @ dd).unsqueeze(-1)
        h[:, positions, :] = seg - coeff * dd
        return h
    return fn


@torch.no_grad()
def logit_lens(model, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the final norm + unembedding to an intermediate hidden state.
    hidden: (..., dim) -> logits: (..., vocab). This is what the model's OWN readout
    pathway would produce from that layer ("prediction" decodability)."""
    return lm_head(model)(final_norm(model)(hidden))
