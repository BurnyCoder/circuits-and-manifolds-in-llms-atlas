"""Environment setup: load ``.env`` before any Hugging Face / torch import, pick the device, seed RNGs.

Local context: ``HF_HOME`` and ``TORCH_HOME`` must be set *before* ``transformers``/``huggingface_hub`` are imported,
because those libraries read the cache location at import time (https://huggingface.co/docs/huggingface_hub/
package_reference/environment_variables). ``setup()`` is therefore called first thing in ``atlas.run``.
Global context: every entry module receives the device chosen here via its ``Ctx``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

from dotenv import load_dotenv

# Repository root = three levels up from this file (src/atlas/common/env.py).
ROOT = Path(__file__).resolve().parents[3]
SITE_DATA = ROOT / "site" / "public" / "data"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"


def setup() -> None:
    """Load ``.env`` (secrets + cache paths) without overriding variables already set in the shell."""
    load_dotenv(ROOT / ".env", override=False)
    # Silence the tokenizers fork warning that appears when DataLoaders spawn workers.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def device() -> str:
    """'cuda' if a GPU is visible, else 'cpu'. Imported lazily so ``setup()`` runs before torch."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU + CUDA) RNGs for reproducible runs."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
