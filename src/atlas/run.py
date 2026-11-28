"""Single entry point: ``uv run python -m atlas.run --entries induction_heads,ioi`` (or ``all``).

Global context: each name maps to a module ``atlas.entries.<name>`` exposing ``compute(ctx)``. Modules are imported
lazily so one entry's heavy dependencies (diffusers, gymnasium, circuit-tracer...) don't load for another.
Phases per entry: load model/data -> compute metrics (+ controls) -> export JSON via ``atlas.common.io.write_entry``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from dataclasses import dataclass, field

from .common.env import device, setup

# Ordered registry: run order for `--entries all` (cheap/independent first).
ENTRIES = [
    "induction_heads",
    "ioi",
    "copy_suppression",
    "greater_than",
    "grokking",
    "grokking_runs",
    "othello_gpt",
    "curve_detectors",
    "calendar_circles",
    "number_helix",
    "linebreak_manifold",
    "belief_fractal",
    "colour_bicone",
    "sae_manifolds",
    "block_sparse",
    "attribution_graphs",
    "neuron_basis",
    "refusal_direction",
    "actadd",
    "jlens",
    "emotions",
    "manifold_steering",
    "calculator_probes",
    "world_model",
    "stories_in_space",
    "sae_features",
]


@dataclass
class Ctx:
    """Run context passed to every entry's ``compute``."""

    device: str
    quick: bool = False  # smaller sample sizes for smoke tests
    seed: int = 0
    extra: dict = field(default_factory=dict)


def src_dirty() -> bool | None:
    """True if `src/` has uncommitted changes (git status --porcelain -- src), None if git is unavailable."""
    import subprocess

    from .common.env import ROOT

    try:
        out = subprocess.check_output(["git", "status", "--porcelain", "--", "src"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    return bool(out.strip())


def main(argv: list[str] | None = None) -> None:
    setup()  # must precede torch/transformers imports (cache paths from .env)
    from .common.log import get_logger

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entries", default="all", help="comma-separated entry module names, or 'all'")
    p.add_argument("--quick", action="store_true", help="small sample sizes (smoke test)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override an entry setting (read by modules from ctx.extra), e.g. --set helix_models=EleutherAI/gpt-j-6b")
    a = p.parse_args(argv)
    if a.quick:
        os.environ["ATLAS_QUICK"] = "1"  # write_entry marks outputs as smoke-test data; atlas.report skips them
    # Audit trail (methods review, m11): record whether the analysis code had uncommitted changes when the run started,
    # so a result can be matched to the exact committed code. write_entry copies this flag into each result's meta.
    os.environ["ATLAS_SRC_DIRTY"] = "1" if src_dirty() else "0"
    names = ENTRIES if a.entries == "all" else [s.strip() for s in a.entries.split(",") if s.strip()]
    log = get_logger("run_" + ("all" if a.entries == "all" else "_".join(names))[:60])
    ctx = Ctx(device=a.device or device(), quick=a.quick, seed=a.seed, extra=dict(kv.split("=", 1) for kv in a.set))
    log.info("device=%s quick=%s seed=%s entries=%s overrides=%s", ctx.device, ctx.quick, ctx.seed, names, ctx.extra)
    failed = []
    for name in names:
        t0 = time.time()
        log.info("=== %s: start", name)
        try:
            mod = importlib.import_module(f"atlas.entries.{name}")
            mod.compute(ctx)
            log.info("=== %s: done in %.1fs", name, time.time() - t0)
        except Exception:
            # Keep going so one smoke-test pass surfaces every broken entry; the traceback goes to the log file.
            log.exception("=== %s: FAILED after %.1fs", name, time.time() - t0)
            failed.append(name)
        finally:
            try:
                free_gpu()
            except Exception:  # a CUDA error in the entry can resurface here; it must not end the whole run
                log.exception("free_gpu after %s failed", name)
    log.info("finished: %d ok, %d failed %s", len(names) - len(failed), len(failed), failed)
    if failed:
        raise SystemExit(1)


def free_gpu() -> None:
    """Release cached CUDA memory between entries (models are loaded per entry; 8 GB VRAM is tight)."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
