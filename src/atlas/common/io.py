"""Export helpers: compact JSON for the website + raw metrics and replication verdicts for ``results/``.

Local context: floats are rounded (``ndigits``) so site data stays small; NumPy/torch values are converted to plain
Python. Global context: the site reads ``site/public/data/<file>.json`` (schema mirrored in site/src/lib/types.ts:
``meta``, ``summary`` for {{templates}}, optional ``replication`` rows, then entry-specific payload) and
``atlas.report`` aggregates ``results/<slug>/replication.json`` into README/report tables.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .env import RESULTS, ROOT, SITE_DATA


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def to_jsonable(x: Any, ndigits: int = 4) -> Any:
    """Recursively convert NumPy/torch/tuples to JSON types, rounding floats to ``ndigits`` significant decimals."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().numpy()
    except ImportError:
        pass
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (np.floating, float)):
        f = float(x)
        if not np.isfinite(f):
            return None
        return round(f, ndigits) if abs(f) >= 1e-3 or f == 0 else float(f"{f:.3g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v, ndigits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v, ndigits) for v in x]
    return x


def verdict_row(claim: str, published: str, ours: str, criterion: str, passed: bool | None, partial: bool = False) -> dict:
    """One row of the published-vs-ours table. ``passed=None`` -> 'n/a'."""
    v = "n/a" if passed is None else ("pass" if passed else ("partial" if partial else "fail"))
    return {"claim": claim, "published": published, "ours": ours, "criterion": criterion, "verdict": v}


def write_entry(
    slug: str,
    payload: dict,
    summary: dict,
    replication: list[dict] | None = None,
    models: list[str] | None = None,
    raw: dict | None = None,
    runtime_s: float | None = None,
    ndigits: int = 4,
) -> None:
    """Write ``site/public/data/<slug>.json`` and ``results/<slug>/{metrics,replication}.json``."""
    meta = {
        "slug": slug,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_commit": _git_commit(),
        "models": models or [],
        "runtime_s": runtime_s,
        # Smoke-test (--quick) outputs are flagged so atlas.report never puts them in result tables.
        "quick": os.environ.get("ATLAS_QUICK") == "1",
    }
    site = {"meta": meta, "summary": summary, "replication": replication or [], **payload}
    SITE_DATA.mkdir(parents=True, exist_ok=True)
    (SITE_DATA / f"{slug}.json").write_text(json.dumps(to_jsonable(site, ndigits), separators=(",", ":")), encoding="utf-8")
    out = RESULTS / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(
        json.dumps(to_jsonable({"meta": meta, "summary": summary, **(raw or {})}, 6), indent=1), encoding="utf-8"
    )
    (out / "replication.json").write_text(json.dumps(to_jsonable(replication or [], 6), indent=1), encoding="utf-8")
