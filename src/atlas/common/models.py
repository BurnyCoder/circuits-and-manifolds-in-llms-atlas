"""Cached model loaders.

Local context: TransformerLens ``HookedTransformer.from_pretrained`` (TL 3.x API, docs:
https://transformerlensorg.github.io/TransformerLens/generated/code/transformer_lens.HookedTransformer.html) gives
hook points for every activation (``blocks.{l}.attn.hook_pattern``, ``hook_z``, ``hook_resid_*``). We cache loaded
models per process so several entries in one ``atlas.run`` reuse GPT-2 small.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def hooked(name: str, device: str = "cuda", **kwargs):
    """Load a TransformerLens HookedTransformer once per (name, device)."""
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(name, device=device, **kwargs)
    model.eval()
    return model
