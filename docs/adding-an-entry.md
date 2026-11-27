# Adding an entry

1. **Pre-state the criteria.** Add the claims, published values, controls and pass criteria to
   [`reports/planned-runs.md`](../reports/planned-runs.md) and commit before running anything.
2. **Write the module.** Create `src/atlas/entries/<module>.py` with a docstring that says what is recomputed, from which
   paper (with link), and how. Implement `compute(ctx)`:
   - Load the model with a cached loader (`atlas.common.models.hooked` for TransformerLens models).
   - Compute the metrics and **controls** (random heads/directions, shuffled labels, random-init models). Use
     `atlas.common.stats.bootstrap_ci` for 95% CIs.
   - Build one `verdict_row(claim, published, ours, criterion, passed)` per pre-stated claim.
   - Call `write_entry(slug, payload, summary, replication, models=[...], runtime_s=...)`. `summary` holds the scalars
     the page templates use; `payload` holds whatever the visualization needs (keep it small, ideally < 1 MB).
   - Respect `ctx.quick` (small samples) so the smoke test finishes in a minute or two.
3. **Register** the module name in `ENTRIES` in `src/atlas/run.py`.
4. **Test.** Add synthetic ground-truth tests for any new math in `tests/`, then run:
   `uv run python -m atlas.run --entries <module> --quick`, then a full run.
5. **Content.** Create `site/src/content/entries/<slug>.md` with frontmatter (see `EntryMeta` in
   `site/src/lib/types.ts`) and the sections *In plain words*, *How it works (technical)*, *What we found*,
   *Limitations*. Every number comes from `{{templates}}`; every claim from the literature gets a citation.
6. **Visualization.** Add `site/src/viz/entries/<Name>.tsx` and register its key in `site/src/viz/registry.ts`. Optionally
   add a schematic thumbnail in `site/src/viz/thumbs.tsx` and `locations` for the Model Explorer.
7. **Report.** Write `reports/experiments/<slug>/experiment.md` (with empty `<!-- table:replication -->` and
   `<!-- table:summary -->` blocks) and `learnings.md`, then run `uv run python -m atlas.report`.
8. **Check.** `npm --prefix site test`, `npm --prefix site run build`, `uv run pytest`; open the page in the dev server.
