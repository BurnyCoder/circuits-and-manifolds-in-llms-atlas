"""Which verdict rows are post-hoc, and why — one auditable list instead of flags scattered across modules.

Local context: a row is *post-hoc* when the analysis behind it (metric, layer choice, selection rule or criterion) was
defined or changed after results for that claim had been seen, in a smoke test or a full run. Fixes of demonstrable
bugs that do not depend on the outcome (tokenization, attention-sink outliers, train/test leakage, out-of-memory) do not
make a row post-hoc; they are listed as amendments in reports/planned-runs.md. Post-hoc rows stay visible everywhere but
are excluded from the pre-stated tally (atlas.report, the site's Replications page).
Global context: applied to every entry's replication rows by common.io.write_entry; `python -m atlas.common.post_hoc`
re-applies it to existing results without re-running models. Added 2026-11-28 after the independent methods review
(reports/reviews/methods-review.md).
"""

from __future__ import annotations

import json

# slug -> {claim prefix -> reason}
POST_HOC: dict[str, dict[str, str]] = {
    "attribution-graphs": {
        "Two-hop": "criteria written after three runs had moved this result from 0/3 to 3/3",
        "Multilingual": "criteria written after the runs had been seen",
    },
    "sae-manifolds": {
        "Subspace capture": "metric switched to restricted R² after a full run whose log showed its peak",
        "Sparsity moves": "claim restated (direction corrected) after a full run",
    },
    "manifold-steering": {
        "Steering along": "layers 22–26 chosen after failing full-mode runs at layers 12–20",
        "Manifold paths": "layers 22–26 chosen after failing full-mode runs at layers 12–20",
    },
    "sae-features": {
        "Adding an SAE": "feature-selection rule changed twice after smoke tests that showed steering outcomes",
        "Feature steering": "feature-selection rule changed twice after smoke tests that showed steering outcomes",
    },
    "calculator-probes": {
        "Base-10 periods stand out": "peak-score analysis added after the pre-stated ranking failed",
    },
    "jlens": {
        "The J-lens surfaces": "arithmetic case removed from the metric after its ranks had been seen",
        "Swapping J-space": "paper-style swap added after the pre-stated swap failed",
    },
    "stories-in-space": {
        "Behaviour and activation": "RSA replaced the centroid analysis after the smoke test",
    },
    "refusal-direction": {
        "Control: a label-shuffled": "stronger control added after the methods review",
    },
    "world-model": {
        "Control: the isometry": "control added after the methods review",
    },
    "block-sparse-featurizers": {
        "Control: random mixed3b": "control added after the methods review",
    },
    "calendar-circles": {
        "The circle is causally": "day-arithmetic templates changed after the smoke test showed this test failing",
    },
}


def mark(slug: str, rows: list[dict]) -> list[dict]:
    """Set row['post_hoc'] = reason for every row of `slug` whose claim starts with a listed prefix (in place)."""
    for row in rows:
        for prefix, why in POST_HOC.get(slug, {}).items():
            if row.get("claim", "").startswith(prefix):
                row["post_hoc"] = why
    return rows


def main() -> None:
    """Re-apply the list to results/<slug>/replication.json and site/public/data/<slug>.json (no model re-runs)."""
    from .env import RESULTS, SITE_DATA

    for slug in POST_HOC:
        rep = RESULTS / slug / "replication.json"
        rows = mark(slug, json.loads(rep.read_text(encoding="utf-8")))
        rep.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        site = SITE_DATA / f"{slug}.json"
        d = json.loads(site.read_text(encoding="utf-8"))
        mark(slug, d.get("replication", []))
        site.write_text(json.dumps(d, separators=(",", ":")), encoding="utf-8")
        print(slug, sum(1 for r in rows if r.get("post_hoc")), "post-hoc rows")


if __name__ == "__main__":
    main()
