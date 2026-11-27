"""Attribution graphs on an open model: the "Biology of an LLM" case studies, re-rendered from public Gemma-2-2B graphs.

Method: Ameisen et al. 2025, "Circuit Tracing" (https://transformer-circuits.pub/2025/attribution-graphs/methods.html) —
replace MLPs by transcoder features, freeze attention, and draw the graph of direct effects between features, token
embeddings, error nodes and output logits on one prompt. Case studies (Claude 3.5 Haiku): Lindsey et al. 2025, "On the
Biology of a Large Language Model" (https://transformer-circuits.pub/2025/attribution-graphs/biology.html).
Open replication: circuit-tracer (Hanna & Piotrowski with Anthropic, MIT) + Neuronpedia, which host *featured* Gemma-2-2B
graphs (GemmaScope transcoders) for the same prompts, with human-labelled supernodes. We fetch those public graphs
read-only (GET /api/graph/{model}/{slug} -> graph JSON; no graphs are generated or uploaded by us), aggregate them to
their labelled supernodes, and run simple structural checks:
- two-hop factual recall: is there a positive path Dallas → "Texas" supernode → "Austin" (and a direct shortcut)?
- multilingual: do the English / French / Chinese antonym graphs share features, more in middle than early/late layers?
Graph data © Neuronpedia / circuit-tracer authors (no explicit data license; attributed and linked; re-rendered summaries only).
"""

from __future__ import annotations

import json
import time
from collections import defaultdict

import numpy as np
import requests

from ..common.env import RESULTS
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "attribution-graphs"
MODEL = "gemma-2-2b"
CASES = {
    "gemma-fact-dallas-austin": "Two-hop fact: Dallas → Texas → Austin",
    "gemma-fact-oakland-sacramento": "Two-hop fact: Oakland → California → Sacramento",
    "gemma-fact-shanghai-beijing": "Two-hop fact: Shanghai → China → Beijing",
    "gemma-small-big-en": "Antonym (English): small → big",
    "gemma-small-big-fr": "Antonym (French): petit → grand",
    "gemma-small-big-zh": "Antonym (Chinese): 小 → 大",
    "gemma-it-rabbit-habit": "Poem planning: rabbit / habit",
    "gemma-addition": "Addition",
    "gemma-michael-jordan": "Known entity: Michael Jordan",
}


def fetch(slug: str) -> dict:
    """Graph JSON for a public Neuronpedia graph (cached under results/, gitignored raw/)."""
    cache = RESULTS / SLUG / "raw" / f"{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    meta = requests.get(f"https://www.neuronpedia.org/api/graph/{MODEL}/{slug}", timeout=60).json()
    g = requests.get(meta["url"], timeout=120).json()
    g["_np_meta"] = {k: meta.get(k) for k in ("slug", "sourceSetName", "prompt", "url")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(g), encoding="utf-8")
    return g


_EXPL: dict[str, str] = {}


def explanation(layer: int, index: int, source_set: str) -> str:
    """Neuronpedia's auto-interp label for a transcoder feature (GET /api/feature/{model}/{layer}-{set}/{index})."""
    key = f"{layer}-{source_set}/{index}"
    if key not in _EXPL:
        try:
            d = requests.get(f"https://www.neuronpedia.org/api/feature/{MODEL}/{key}", timeout=30).json()
            ex = [e.get("description", "").strip() for e in d.get("explanations", []) if e.get("description")]
            _EXPL[key] = ex[0] if ex else ""
        except Exception:
            _EXPL[key] = ""
    return _EXPL[key]


def summarize(slug: str, g: dict) -> dict:
    """Aggregate a graph to labelled supernodes + pinned nodes + embeddings + logits; edges summed between groups."""
    nodes = {n["node_id"]: n for n in g["nodes"]}
    q = g.get("qParams", {}) or {}
    groups: dict[str, dict] = {}
    member_of: dict[str, str] = {}
    for sn in q.get("supernodes", []) or []:
        label, members = sn[0], [m for m in sn[1:] if m in nodes]
        if not members:
            continue
        gid = f"S:{label}"
        groups[gid] = {"id": gid, "label": label, "kind": "super", "members": members}
        for m in members:
            member_of[m] = gid
    keep = set(q.get("pinnedIds", []) or [])
    if not groups and not keep:
        # Uncurated graph: keep the 30 most important transcoder features. In circuit-tracer's graph files a node's
        # "influence" is the cumulative-influence level at which it survives pruning, so smaller = more important.
        feats = [n for n in g["nodes"] if n.get("feature_type") == "cross layer transcoder"]
        keep = {n["node_id"] for n in sorted(feats, key=lambda n: n.get("influence") or 1.0)[:30]}
    # Always include target logits and embeddings of all prompt tokens.
    for nid, n in nodes.items():
        if n.get("feature_type") == "logit" and (n.get("is_target_logit") or n.get("token_prob", 0) > 0.05):
            keep.add(nid)
        if n.get("feature_type") == "embedding":
            keep.add(nid)
    for nid in keep:
        if nid in member_of or nid not in nodes:
            continue
        n = nodes[nid]
        ft = n.get("feature_type")
        kind = {"embedding": "embed", "logit": "logit", "mlp reconstruction error": "error"}.get(ft, "feature")
        label = n.get("clerp") or ""
        if kind == "feature" and not label:
            idx = int(n["feature"]) % 100000  # node feature ids encode layer * 100000 + index
            label = explanation(int(n["layer"]), idx, (g.get("_np_meta", {}).get("sourceSetName") or "gemmascope-transcoder-16k")) or f"L{n['layer']} #{idx}"
        elif kind == "error":
            label = f"error node L{n['layer']}"
        groups[nid] = {"id": nid, "label": label, "kind": kind, "members": [nid]}
        member_of[nid] = nid
    tokens = g["metadata"]["prompt_tokens"]
    for gid, grp in groups.items():
        mem = [nodes[m] for m in grp["members"]]
        layers = [(-1 if m["feature_type"] == "embedding" else (99 if m["feature_type"] == "logit" else int(m["layer"]))) for m in mem]
        grp["layer"] = float(np.mean(layers))
        grp["ctx"] = float(np.mean([m["ctx_idx"] for m in mem]))
        grp["influence"] = float(np.mean([m.get("influence") or 0 for m in mem]))
        if grp["kind"] == "embed":
            grp["label"] = tokens[mem[0]["ctx_idx"]]
        if grp["kind"] == "logit":
            grp["label"] = f"→ {mem[0].get('clerp') or ''}".strip()
            grp["prob"] = float(mem[0].get("token_prob") or 0)
    w = defaultdict(float)
    for link in g["links"]:
        a, b = member_of.get(link["source"]), member_of.get(link["target"])
        if a and b and a != b:
            w[(a, b)] += link["weight"]
    edges = [{"s": a, "t": b, "w": v} for (a, b), v in w.items()]
    # Keep, per target, incoming edges with ≥ 5% of that target's strongest incoming |weight| (a global cutoff would
    # delete the weaker but essential edges into the output logits).
    max_in = defaultdict(float)
    for e in edges:
        max_in[e["t"]] = max(max_in[e["t"]], abs(e["w"]))
    edges = [e for e in edges if abs(e["w"]) >= 0.05 * max_in[e["t"]]]
    # Error-node share of total incoming |weight| into logits ("dark matter" of the replacement model).
    logit_ids = {nid for nid, n in nodes.items() if n.get("feature_type") == "logit"}
    into_logits = [l for l in g["links"] if l["target"] in logit_ids]
    err = sum(abs(l["weight"]) for l in into_logits if nodes.get(l["source"], {}).get("feature_type") == "mlp reconstruction error")
    tot = sum(abs(l["weight"]) for l in into_logits) or 1.0
    feats = {f"{n['layer']}_{n['feature']}": int(n["layer"]) for n in g["nodes"] if n.get("feature_type") == "cross layer transcoder"}
    return {
        "slug": slug,
        "title": CASES[slug],
        "prompt": g["metadata"]["prompt"],
        "tokens": tokens,
        "nodes": list(groups.values()),
        "edges": edges,
        "n_nodes": len(g["nodes"]),
        "n_links": len(g["links"]),
        "error_share_into_logits": err / tot,
        "source_set": g.get("_np_meta", {}).get("sourceSetName"),
        "np_url": f"https://www.neuronpedia.org/{MODEL}/graph?slug={slug}",
        "_feats": feats,
    }


def path_strength(s: dict, start_label: str, mid_label: str, end_label: str) -> tuple[float, float]:
    """Best product of positive edge weights start→mid→end, and the direct start→end weight (group labels, substring match)."""
    by = {n["id"]: n for n in s["nodes"]}
    W = {(e["s"], e["t"]): e["w"] for e in s["edges"]}
    starts = [n["id"] for n in s["nodes"] if start_label.lower() in n["label"].lower() and n["kind"] == "embed"]
    mids = [n["id"] for n in s["nodes"] if mid_label.lower() in n["label"].lower() and n["kind"] != "embed"]
    ends = [n["id"] for n in s["nodes"] if end_label.lower() in n["label"].lower() and n["kind"] == "logit"]
    best = 0.0
    for a in starts:
        for m in mids:
            for b in ends:
                # two-hop through m, allowing one intermediate group between a and m
                direct_am = W.get((a, m), 0)
                via = max([min(W.get((a, x), 0), W.get((x, m), 0)) for x in by] + [direct_am])
                best = max(best, min(via, max(W.get((m, b), 0), max([min(W.get((m, y), 0), W.get((y, b), 0)) for y in by]))))
    direct = max([W.get((a, b), 0) for a in starts for b in ends] + [0.0])
    return best, direct


def compute(ctx) -> None:
    t0 = time.time()
    slugs = list(CASES)[:3] if ctx.quick else list(CASES)
    graphs = []
    for slug in slugs:
        try:
            graphs.append(summarize(slug, fetch(slug)))
            log.info("graph %s: %d nodes, %d links, %d groups, error share %.2f", slug, graphs[-1]["n_nodes"], graphs[-1]["n_links"], len(graphs[-1]["nodes"]), graphs[-1]["error_share_into_logits"])
        except Exception as e:  # network or format problems are logged, the entry still builds
            log.warning("graph %s failed: %s", slug, e)
    by = {g["slug"]: g for g in graphs}
    two_hop = []
    for slug, (a, m, b) in {
        "gemma-fact-dallas-austin": ("Dallas", "Texas", "Austin"),
        "gemma-fact-oakland-sacramento": ("Oakland", "California", "Sacramento"),
        "gemma-fact-shanghai-beijing": ("Shanghai", "China", "Beijing"),
    }.items():
        if slug in by:
            ph, d = path_strength(by[slug], a, m, b)
            two_hop.append({"slug": slug, "path": ph, "direct": d})
            log.info("%s: two-hop path strength %.3f, direct %.3f", slug, ph, d)
    # Multilingual feature overlap by layer band.
    overlap = {}
    langs = [s for s in ("gemma-small-big-en", "gemma-small-big-fr", "gemma-small-big-zh") if s in by]
    if len(langs) == 3:
        sets = [set(by[s]["_feats"]) for s in langs]
        layer_of = {**by[langs[0]]["_feats"], **by[langs[1]]["_feats"], **by[langs[2]]["_feats"]}
        for band, (lo, hi) in {"early (0–7)": (0, 7), "middle (8–17)": (8, 17), "late (18–25)": (18, 25)}.items():
            ss = [{f for f in s if lo <= layer_of[f] <= hi} for s in sets]
            inter = len(ss[0] & ss[1] & ss[2])
            union = len(ss[0] | ss[1] | ss[2]) or 1
            overlap[band] = inter / union
        log.info("multilingual feature overlap (Jaccard of 3 graphs): %s", overlap)
    for g in graphs:
        g.pop("_feats")
    n_paths = sum(1 for t in two_hop if t["path"] > 0)
    summary = {
        "n_graphs": len(graphs),
        "two_hop_found": n_paths,
        "two_hop_total": len(two_hop),
        "overlap_early": overlap.get("early (0–7)", -1),
        "overlap_middle": overlap.get("middle (8–17)", -1),
        "overlap_late": overlap.get("late (18–25)", -1),
        "error_share_median": float(np.median([g["error_share_into_logits"] for g in graphs])) if graphs else -1,
    }
    repl = [
        verdict_row(
            "Two-hop factual recall goes through an intermediate concept",
            "Dallas → Texas → Austin path plus a direct shortcut (Claude 3.5 Haiku)",
            f"positive embedding → intermediate → answer path in {n_paths}/{len(two_hop)} public Gemma-2-2B graphs (Dallas, Oakland, Shanghai)",
            "path present in all checked graphs",
            n_paths == len(two_hop) and len(two_hop) > 0,
            partial=n_paths > 0,
        ),
        verdict_row(
            "Multilingual circuits share language-independent features in the middle",
            "20 of 27 features shared across English/French/Chinese; language-specific features early and late (Claude)",
            f"feature overlap (Jaccard of the 3 graphs): early {summary['overlap_early']:.2f}, middle {summary['overlap_middle']:.2f}, late {summary['overlap_late']:.2f}",
            "middle-layer overlap higher than both early and late",
            summary["overlap_middle"] > max(summary["overlap_early"], summary["overlap_late"]),
        ),
    ]
    write_entry(SLUG, {"graphs": graphs, "two_hop": two_hop, "overlap": overlap, "model": MODEL}, summary, repl, models=["gemma-2-2b (Neuronpedia public graphs)"], runtime_s=time.time() - t0, ndigits=3)
    log.info("attribution graphs summary: %s", summary)
