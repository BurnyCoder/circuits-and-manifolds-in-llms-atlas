"""Single entry point: ``uv run python -m atlas.run --entries induction_heads,ioi`` (or ``all``).

Global context: each name maps to a module ``atlas.entries.<name>`` exposing ``compute(ctx)``. Modules are imported
lazily so one entry's heavy dependencies (diffusers, gymnasium, circuit-tracer...) don't load for another.
Phases per entry: load model/data -> compute metrics (+ controls) -> export JSON via ``atlas.common.io.write_entry``.
"""

from __future__ import annotations

import argparse
import importlib
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
    "othello_gpt",
    "curve_detectors",
]


@dataclass
class Ctx:
    """Run context passed to every entry's ``compute``."""

    device: str
    quick: bool = False  # smaller sample sizes for smoke tests
    seed: int = 0
    extra: dict = field(default_factory=dict)


def main(argv: list[str] | None = None) -> None:
    setup()  # must precede torch/transformers imports (cache paths from .env)
    from .common.log import get_logger

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--entries", default="all", help="comma-separated entry module names, or 'all'")
    p.add_argument("--quick", action="store_true", help="small sample sizes (smoke test)")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    names = ENTRIES if a.entries == "all" else [s.strip() for s in a.entries.split(",") if s.strip()]
    log = get_logger("run_" + ("all" if a.entries == "all" else "_".join(names))[:60])
    ctx = Ctx(device=a.device or device(), quick=a.quick, seed=a.seed)
    log.info("device=%s quick=%s seed=%s entries=%s", ctx.device, ctx.quick, ctx.seed, names)
    for name in names:
        t0 = time.time()
        log.info("=== %s: start", name)
        mod = importlib.import_module(f"atlas.entries.{name}")
        mod.compute(ctx)
        log.info("=== %s: done in %.1fs", name, time.time() - t0)


if __name__ == "__main__":
    main()
