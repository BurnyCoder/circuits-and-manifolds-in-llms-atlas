"""Upload the models trained in this repository to Hugging Face, each with a model card, into one collection.

Local context: one model repo per trained model family (state_dict ``.pt`` files + ``config.json`` + README model card),
then all repos are added to the collection "Circuits & Manifolds Atlas" under the user's namespace. Uses
``huggingface_hub.HfApi`` (create_repo / upload_folder / create_collection / add_collection_item; docs:
https://huggingface.co/docs/huggingface_hub/guides/upload and .../guides/collections). Re-running is idempotent
(``exist_ok`` / ``exists_ok``). Global context: CLAUDE.md asks for one HF collection per research project; the
checkpoints are written by ``atlas.entries.{grokking,belief_fractal,world_model}`` and are gitignored.

Usage: ``uv run python -m atlas.hf_upload [--only grokking,belief,world-model]``
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from .common.env import RESULTS, setup

REPO_URL = "https://github.com/BurnyCoder/circuits-and-manifolds-in-llms-atlas"
SITE_URL = "https://burnycoder.github.io/circuits-and-manifolds-in-llms-atlas/#/entry/"
COLLECTION = "Circuits & Manifolds Atlas"
MODULE = {"grokking": "grokking", "belief": "belief_fractal", "world-model": "world_model"}  # code that defines each model

# name -> (repo suffix, checkpoint glob in results/, entry slug, config dict, card body)
MODELS = {
    "grokking": (
        "atlas-grokking-modadd-p113",
        ("grokking", "modadd_p113_seed*.pt"),
        "grokking",
        {"task": "(a + b) mod 113", "n_layers": 1, "d_model": 128, "n_heads": 4, "d_head": 32, "d_mlp": 512,
         "act_fn": "relu", "normalization": None, "d_vocab": 114, "d_vocab_out": 113, "n_ctx": 3, "train_fraction": 0.3,
         "optimizer": "AdamW lr 1e-3, weight decay 1.0, betas (0.9, 0.98), full batch", "epochs": 40000,
         "seeds": [0, 1, 2], "framework": "TransformerLens HookedTransformer state_dict"},
        "One-layer transformers trained from scratch on modular addition with the settings of Nanda et al. 2023, "
        "\"Progress measures for grokking via mechanistic interpretability\" (https://arxiv.org/abs/2301.05217). All three "
        "seeds memorize within a few hundred epochs and generalize thousands of epochs later, using a few key Fourier "
        "frequencies (see the atlas entry for curves, frequencies and ablations).",
    ),
    "belief": (
        "atlas-mess3-belief-transformer",
        ("belief-fractal", "mess3_transformer.pt"),
        "belief-fractal",
        {"task": "next-token prediction on the Mess3 hidden Markov process (x = 0.05, alpha = 0.85)", "n_layers": 4,
         "d_model": 64, "n_heads": 1, "d_head": 64, "d_mlp": 256, "act_fn": "relu", "normalization": "LN",
         "d_vocab": 3, "n_ctx": 10, "optimizer": "Adam lr 1e-3, 512 sequences per step", "steps": 30000,
         "framework": "TransformerLens HookedTransformer state_dict"},
        "A small transformer trained on sequences from the Mess3 process, following Shai et al. 2024, \"Transformers "
        "represent belief state geometry in their residual stream\" (https://arxiv.org/abs/2405.15943). Its final residual "
        "stream linearly encodes the Bayes-optimal belief over hidden states (a fractal on the 2-simplex).",
    ),
    "world-model": (
        "atlas-mountaincar-world-model",
        ("world-model", "mountain_car_world_model.pt"),
        "world-model",
        {"task": "next-frame prediction on gymnasium MountainCar-v0 rollouts",
         "architecture": "conv encoder (64x64 frames -> 64-d LayerNorm latent), 16-d action embedding, GRU-64, conv decoders (next and current frame)",
         "framework": "PyTorch state_dict (class definition in src/atlas/entries/world_model.py)"},
        "A small world model trained on Mountain Car frames, used to replicate the position-manifold analysis of "
        "Goodfire's \"The World Inside Neural Networks\" (https://www.goodfire.com/research/the-world-inside-neural-networks) "
        "and Wurgaft et al. (https://arxiv.org/abs/2605.05115, App. B): car position lies on a curved 1-D manifold in the latent.",
    ),
}


def card(name: str, repo_id: str, files: list[str]) -> str:
    """Model card (YAML metadata + description + files + how to load + links)."""
    _, _, slug, cfg, body = MODELS[name]
    return f"""---
license: mit
library_name: pytorch
tags: [interpretability, mechanistic-interpretability, circuits-and-manifolds-atlas]
---

# {repo_id.split('/')[-1]}

{body}

Part of the **Atlas of Circuits & Manifolds** — code: {REPO_URL} · interactive entry: {SITE_URL}{slug}

## Files

{chr(10).join(f"- `{f}`" for f in files)}
- `config.json` — architecture and training settings

## Configuration

```json
{json.dumps(cfg, indent=2)}
```

## How to load

The checkpoints are plain PyTorch `state_dict`s. Rebuild the model with the code in
`src/atlas/entries/{MODULE[name]}.py` of the repository above,
then `model.load_state_dict(torch.load(path, map_location="cpu"))`.

## Limitations

Research artifacts for interpretability replications; tiny models trained on synthetic tasks, not for any downstream use.
"""


def main(argv: list[str] | None = None) -> None:
    setup()
    from huggingface_hub import HfApi

    p = argparse.ArgumentParser()
    p.add_argument("--only", default=",".join(MODELS))
    a = p.parse_args(argv)
    api = HfApi()
    user = api.whoami()["name"]
    collection = api.create_collection(COLLECTION, namespace=user, exists_ok=True,
                                       description="Models trained for the Atlas of Circuits & Manifolds (interactive, cited replications).")
    for name in a.only.split(","):
        suffix, (folder, pattern), slug, cfg, _ = MODELS[name]
        ckpts = sorted((RESULTS / folder).glob(pattern))
        if not ckpts:
            print(f"skip {name}: no checkpoints matching results/{folder}/{pattern} (run the entry first)")
            continue
        repo_id = f"{user}/{suffix}"
        api.create_repo(repo_id, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for c in ckpts:
                shutil.copy(c, tmp / c.name)
            (tmp / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            (tmp / "README.md").write_text(card(name, repo_id, [c.name for c in ckpts]), encoding="utf-8")
            api.upload_folder(repo_id=repo_id, folder_path=tmp, commit_message=f"Upload {name} checkpoints from the atlas pipeline")
        api.add_collection_item(collection.slug, repo_id, "model", exists_ok=True, note=f"Atlas entry: {SITE_URL}{slug}")
        print(f"uploaded {repo_id} ({len(ckpts)} checkpoints)")
    print("collection:", f"https://huggingface.co/collections/{collection.slug}")


if __name__ == "__main__":
    main()
