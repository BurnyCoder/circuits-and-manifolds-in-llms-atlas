"""Timestamped logging to the terminal and to ``logs/<timestamp>_<name>.log``.

Local context: one log file per ``atlas.run`` invocation; every entry logs its parameters, metrics and (for
generation entries) the full prompts and outputs, untruncated, as the user's instructions require.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime

from .env import LOGS

_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(run_name: str = "atlas") -> logging.Logger:
    """Return the root 'atlas' logger, attaching console + file handlers on first call."""
    log = logging.getLogger("atlas")
    if log.handlers:
        return log
    log.setLevel(logging.INFO)
    log.propagate = False  # libraries may install a root handler (basicConfig); avoid printing every line twice
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / f"{datetime.now():%Y%m%d_%H%M%S}_{run_name}.log"
    # Windows consoles/pipes default to cp1252, which cannot print α, ², arrows or non-Latin text in model outputs
    # ("'charmap' codec can't encode ..."); force UTF-8 so terminal logs are complete like the file log.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        h.setFormatter(logging.Formatter(_FMT, datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(h)
    log.info("logging to %s", path)
    return log
