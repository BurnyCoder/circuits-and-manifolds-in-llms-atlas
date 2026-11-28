"""Regenerate every results table in README.md and reports/ from ``results/<slug>/*.json``.

Local context: prose files carry marker pairs ``<!-- table:NAME -->`` … ``<!-- /table:NAME -->``; this script
rewrites only the text between them, so hand-written explanations survive and every number in a table comes from a
results file (CLAUDE.md: "no hand-typed numbers, and one command regenerates all tables").
Global context: entry modules write ``results/<slug>/{metrics,replication}.json`` via ``common.io.write_entry``; the
website reads the same verdict rows from ``site/public/data/<slug>.json``.

Usage: ``uv run python -m atlas.report``
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .common.env import RESULTS, ROOT

EXPERIMENTS = ROOT / "reports" / "experiments"
# Files whose `scoreboard` table lists every recomputed entry.
SCOREBOARD_FILES = [ROOT / "README.md", EXPERIMENTS / "experiments.md"]
VERDICT_ICON = {"pass": "✅ pass", "partial": "🟡 partial", "fail": "❌ fail", "n/a": "n/a"}


def load_results() -> dict[str, dict]:
    """Map slug -> {'meta', 'summary', 'replication'} for every results/<slug>/ folder with verdict rows.

    Support files without claims (e.g. ``gpt2-map``, the Model Explorer's head scores) are skipped.
    """
    out = {}
    for d in sorted(p for p in RESULTS.iterdir() if (p / "metrics.json").exists()):
        metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        rep_file = d / "replication.json"
        rep = json.loads(rep_file.read_text(encoding="utf-8")) if rep_file.exists() else []
        if not rep or metrics.get("meta", {}).get("quick"):
            continue  # no claims, or a --quick smoke-test output
        out[d.name] = {"meta": metrics.get("meta", {}), "summary": metrics.get("summary", {}), "replication": rep}
    return out


def _cell(text: object) -> str:
    """Make a value safe inside a Markdown table cell (pipes would split the cell)."""
    return str(text).replace("|", "\\|").replace("\n", " ")


def scoreboard_table(res: dict[str, dict], from_dir: Path = ROOT) -> str:
    """One row per entry: models, claims, pass / partial / fail counts, runtime, link to its report.

    Links are written relative to ``from_dir`` (the directory of the file the table goes into).
    """
    lines = ["| Entry | Models | Pre-stated claims | ✅ pass | 🟡 partial | ❌ fail | Post-hoc rows (not counted) | Runtime (s) | Generated (UTC) |",
             "|---|---|---|---|---|---|---|---|---|"]
    totals = {"pass": 0, "partial": 0, "fail": 0, "claims": 0, "post_hoc": 0}
    for slug, r in res.items():
        # Post-hoc rows (analysis chosen after seeing results for that claim) are listed but never counted.
        pre = [row for row in r["replication"] if not row.get("post_hoc")]
        n_post = len(r["replication"]) - len(pre)
        verdicts = [row.get("verdict", "n/a") for row in pre]
        counts = {k: verdicts.count(k) for k in ("pass", "partial", "fail")}
        for k, v in counts.items():
            totals[k] += v
        totals["claims"] += len(verdicts)
        totals["post_hoc"] += n_post
        meta = r["meta"]
        runtime = meta.get("runtime_s")
        link = Path(os.path.relpath(EXPERIMENTS / slug / "experiment.md", from_dir)).as_posix()
        lines.append(
            f"| [{slug}]({link}) | {_cell(', '.join(meta.get('models', [])))} | "
            f"{len(verdicts)} | {counts['pass']} | {counts['partial']} | {counts['fail']} | {n_post} | "
            f"{'' if runtime is None else round(runtime)} | {meta.get('generated', '')} |"
        )
    lines.append(f"| **Total ({len(res)} entries)** | | **{totals['claims']}** | **{totals['pass']}** | "
                 f"**{totals['partial']}** | **{totals['fail']}** | **{totals['post_hoc']}** | | |")
    return "\n".join(lines)


def verdict_cell(row: dict) -> str:
    """Verdict with icon; post-hoc rows are marked with their reason."""
    v = VERDICT_ICON.get(row.get("verdict", "n/a"), row.get("verdict"))
    return f"{v} (post-hoc: {_cell(row['post_hoc'])})" if row.get("post_hoc") else v


def headline_table(res: dict[str, dict]) -> str:
    """One row per entry: its first (headline) claim, the published value, ours, and the verdict."""
    lines = ["| Entry | Headline claim | Published | Ours | Verdict |", "|---|---|---|---|---|"]
    for slug, r in res.items():
        row = r["replication"][0]
        lines.append(f"| {slug} | " + " | ".join(_cell(row.get(k, "")) for k in ("claim", "published", "ours"))
                     + f" | {verdict_cell(row)} |")
    return "\n".join(lines)


CONTENT = ROOT / "site" / "src" / "content" / "entries"
FIDELITY_ICON = {"recomputed": "🟢 recomputed", "replicated": "🔵 open replication", "redrawn": "🟠 redrawn"}
SITE_URL = "https://burnycoder.github.io/circuits-and-manifolds-in-llms-atlas/#/entry/"


def catalogue_table() -> str:
    """Every atlas entry (from the site's Markdown frontmatter): title, category, models, fidelity, evidence."""
    import yaml  # PyYAML, a transitive dependency of transformers / huggingface_hub

    metas = []
    for f in CONTENT.glob("*.md"):
        front = f.read_text(encoding="utf-8").split("---")[1]
        metas.append(yaml.safe_load(front))
    metas.sort(key=lambda m: m.get("order", 999))
    lines = ["| Entry | Category | Models | Data | Evidence |", "|---|---|---|---|---|"]
    for m in metas:
        lines.append(f"| [{_cell(m['title'])}]({SITE_URL}{m['slug']}) | {m['category']} | {_cell(', '.join(m['models']))} | "
                     f"{FIDELITY_ICON.get(m['fidelity'], m['fidelity'])} | {m['evidence']} |")
    return "\n".join(lines)


def replication_table(rows: list[dict]) -> str:
    """Published claim vs our measurement vs the pre-stated criterion (from reports/planned-runs.md) vs verdict."""
    lines = ["| Claim | Published | Ours | Criterion | Verdict |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(row.get(k, "")) for k in ("claim", "published", "ours", "criterion"))
                     + f" | {verdict_cell(row)} |")
    return "\n".join(lines)


def summary_table(summary: dict) -> str:
    """Every scalar in the entry's summary dict (the same values the website templates read)."""
    lines = ["| Key | Value |", "|---|---|"]
    for k, v in summary.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            lines.append(f"| `{k}` | {_cell(v)} |")
    return "\n".join(lines)


def replace_block(text: str, name: str, body: str) -> str:
    """Replace the content between <!-- table:name --> and <!-- /table:name --> (markers kept)."""
    pattern = re.compile(rf"(<!-- table:{re.escape(name)} -->)(.*?)(<!-- /table:{re.escape(name)} -->)", re.S)
    return pattern.sub(lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text)


def update_file(path: Path, tables: dict[str, str]) -> bool:
    """Rewrite all marker blocks present in ``path``; return True if the file changed."""
    if not path.exists():
        return False
    old = path.read_text(encoding="utf-8")
    new = old
    for name, body in tables.items():
        new = replace_block(new, name, body)
    if new != old:
        path.write_text(new, encoding="utf-8")
    return new != old


def ensure_experiment_file(slug: str) -> Path:
    """Create reports/experiments/<slug>/experiment.md with empty marker blocks if it does not exist yet."""
    path = EXPERIMENTS / slug / "experiment.md"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"# Experiment: {slug}\n\n## Results vs published claims\n\n<!-- table:replication -->\n"
            "<!-- /table:replication -->\n\n## All summary values\n\n<!-- table:summary -->\n<!-- /table:summary -->\n",
            encoding="utf-8",
        )
    return path


def main() -> None:
    res = load_results()
    changed = []
    for f in SCOREBOARD_FILES:
        tables = {"scoreboard": scoreboard_table(res, f.parent), "headlines": headline_table(res), "catalogue": catalogue_table()}
        if update_file(f, tables):
            changed.append(f)
    for slug, r in res.items():
        f = ensure_experiment_file(slug)
        if update_file(f, {"replication": replication_table(r["replication"]), "summary": summary_table(r["summary"])}):
            changed.append(f)
    print(f"{len(res)} entries; updated {len(changed)} files:")
    for f in changed:
        print("  ", f.relative_to(ROOT))


if __name__ == "__main__":
    main()
