"""The character-count manifold behind line breaking — recomputed on Qwen3-1.7B (GPT-2 small as a weak-model control).

Original (closed model): Gurnee, Ameisen, Kauvar, Tarng, Pearce, Olah, Batson 2025, "When Models Manipulate Manifolds:
The Geometry of a Counting Task" (https://transformer-circuits.pub/2025/linebreaks/index.html; arXiv 2601.04480). In
Claude 3.5 Haiku the number of characters since the last newline is encoded early (probe R² 0.985 after layer 1) on a
curved, "rippled" low-dimensional manifold (top 6 PCs of per-count means hold 95% of variance), whose cosine-similarity
matrix "rings" (diagonal band + echoes). Open replication: Sinii & Balagansky 2026 (T-Tech, "Chasing the Counting
Manifold in Open LLMs", Apache-2.0) found linear decodability (Qwen3-8B R² 0.852 at L5) and a helix with period ~155
characters, but not the very-early-layer peak.
We wrap WikiText-103 paragraphs at widths 40–150 characters and, for every token, record the characters since the
last newline. Per layer: ridge-probe R² (held-out documents) vs a shuffled-label control, PCA of per-count mean
activations (variance in 6 PCs, 3-D curve), the cosine "ringing" matrix, and the model's newline behaviour (AUC of
P(newline) for tokens that are actually followed by a line break).
"""

from __future__ import annotations

import textwrap
import time

import numpy as np
import torch

from ..common.geometry import pca, ridge_r2
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "linebreak-manifold"
WIDTHS = [40, 60, 80, 100, 120, 150]
MAXC = 150


def wrapped_docs(n_docs: int, seed: int) -> list[tuple[str, int]]:
    """WikiText-103 paragraphs (≥ 600 chars) hard-wrapped with textwrap at a random width from WIDTHS."""
    from datasets import load_dataset

    paras = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"] if len(t.strip()) > 600 and not t.strip().startswith("=")]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(paras), size=min(n_docs, len(paras)), replace=False)
    out = []
    for i in idx:
        w = int(rng.choice(WIDTHS))
        out.append((textwrap.fill(paras[i][:1500], width=w), w))
    return out


def char_counts(text: str, offsets: list[tuple[int, int]]) -> np.ndarray:
    """Characters since the last newline at the end of each token (0 for tokens containing a newline)."""
    last_nl = -1
    nl_pos = [i for i, ch in enumerate(text) if ch == "\n"]
    counts = []
    j = 0
    for s, e in offsets:
        while j < len(nl_pos) and nl_pos[j] < e:
            last_nl = nl_pos[j]
            j += 1
        counts.append(0 if "\n" in text[s:e] else e - last_nl - 1)
    return np.array(counts)


@torch.no_grad()
def analyze(model_name: str, ctx, n_docs: int) -> dict:
    from sklearn.metrics import roc_auc_score
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16 if "Qwen" in model_name else torch.float32).to(ctx.device).eval()
    nl_ids = {i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t])}
    n_layers = model.config.num_hidden_layers
    probe_layers = sorted(set(range(0, n_layers + 1, 3)) | {1, 2})
    d = model.config.hidden_size
    sums = np.zeros((n_layers + 1, MAXC + 1, d))
    cnt = np.zeros(MAXC + 1)
    samples = {L: [] for L in probe_layers}
    labels, docs_of = [], []
    p_nl, is_nl = [], []
    rng = np.random.default_rng(ctx.seed)
    docs = wrapped_docs(n_docs, ctx.seed)
    for di, (text, w) in enumerate(docs):
        enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, truncation=True, max_length=512)
        ids = torch.tensor([enc.input_ids], device=ctx.device)
        out = model(ids, output_hidden_states=True)
        counts = char_counts(text, enc.offset_mapping)
        hs = torch.stack(out.hidden_states)[:, 0].float().cpu().numpy()  # [n_layers+1, T, d]
        keep = np.where((counts >= 1) & (counts <= MAXC))[0]
        for t in keep:
            sums[:, counts[t]] += hs[:, t]
            cnt[counts[t]] += 1
        sub = keep[rng.random(len(keep)) < 0.07]  # subsample tokens for probes (memory: ~0.5 GB at fp16)
        for L in probe_layers:
            samples[L].append(hs[L, sub].astype(np.float16))
        labels += counts[sub].tolist()
        docs_of += [di] * len(sub)
        probs = out.logits[0, :-1].float().softmax(-1)
        pn = probs[:, list(nl_ids)].sum(-1).cpu().numpy()
        nxt = np.array([i in nl_ids for i in enc.input_ids[1:]])
        p_nl += pn[counts[:-1] > 0].tolist()
        is_nl += nxt[counts[:-1] > 0].tolist()
    labels, docs_of = np.array(labels, dtype=float), np.array(docs_of)
    test_docs = docs_of % 5 == 0  # hold out every 5th document
    probe = []
    shuffled = rng.permutation(labels)
    for L in probe_layers:
        X = np.concatenate(samples[L]).astype(np.float32)
        r2 = ridge_r2(X[~test_docs], labels[~test_docs, None], X[test_docs], labels[test_docs, None], alpha=10.0)
        r2s = ridge_r2(X[~test_docs], shuffled[~test_docs, None], X[test_docs], shuffled[test_docs, None], alpha=10.0)
        probe.append({"layer": L, "r2": r2, "r2_shuffled": r2s})
        log.info("%s L%d count probe R² %.3f (shuffled %.3f)", model_name, L, r2, r2s)
    # Geometry of per-count means at the layer with the best probe (and its neighbours).
    best = max(probe, key=lambda p: p["r2"])["layer"]
    valid = np.where(cnt[1:] >= 20)[0] + 1  # counts with enough tokens
    geo = {}
    for L in sorted({1, best, min(n_layers, best + 4)}):
        M = sums[L, valid] / cnt[valid, None]
        scores, comps, var, mu = pca(M, 6)
        Mc = M - M.mean(0)
        Mn = Mc / np.linalg.norm(Mc, axis=1, keepdims=True)
        cos = Mn @ Mn.T
        ang = np.unwrap(np.arctan2(scores[:, 1], scores[:, 0]))
        slope = float(np.polyfit(valid, np.degrees(ang), 1)[0])
        geo[str(L)] = {"counts": valid, "pcs": scores, "var": var, "cos": cos[::2, ::2], "deg_per_char": slope}
    auc = float(roc_auc_score(is_nl, p_nl)) if any(is_nl) else float("nan")
    log.info("%s: best probe layer %d, newline AUC %.3f", model_name, best, auc)
    del model
    torch.cuda.empty_cache()
    return {"name": model_name, "n_layers": n_layers, "probe": probe, "best_layer": best, "geo": geo, "newline_auc": auc, "n_tokens": int(cnt.sum())}


def compute(ctx) -> None:
    t0 = time.time()
    n = 40 if ctx.quick else 400
    q = analyze("Qwen/Qwen3-1.7B", ctx, n)
    g = analyze("gpt2", ctx, n)
    qb = max(q["probe"], key=lambda p: p["r2"])
    gb = max(g["probe"], key=lambda p: p["r2"])
    qgeo = q["geo"][str(q["best_layer"])]
    early = [p for p in q["probe"] if p["layer"] <= 2]
    summary = {
        "q_best_layer": q["best_layer"],
        "q_best_r2": qb["r2"],
        "q_best_r2_shuffled": qb["r2_shuffled"],
        "q_early_r2": max(p["r2"] for p in early),
        "q_var6": float(qgeo["var"].sum()),
        "q_var3": float(qgeo["var"][:3].sum()),
        "q_deg_per_char": qgeo["deg_per_char"],
        "q_period_chars": 360.0 / abs(qgeo["deg_per_char"]) if qgeo["deg_per_char"] else -1,
        "q_auc": q["newline_auc"],
        "g_best_r2": gb["r2"],
        "g_auc": g["newline_auc"],
        "q_tokens": q["n_tokens"],
    }
    repl = [
        verdict_row(
            "Character count since the last newline is linearly decodable",
            "Haiku: probe R² 0.985 after layer 1; open models (T-Tech): Qwen3-8B 0.852 at L5, Gemma-2-9B 0.868",
            f"Qwen3-1.7B: best held-out R² {qb['r2']:.3f} at layer {q['best_layer']} (shuffled labels {qb['r2_shuffled']:.3f}); GPT-2 small {gb['r2']:.3f}",
            "Qwen3-1.7B best R² ≥ 0.7 and ≥ 0.5 above the shuffled control",
            qb["r2"] >= 0.7 and qb["r2"] - qb["r2_shuffled"] >= 0.5,
        ),
        verdict_row(
            "The counts lie on a low-dimensional curved manifold",
            "Haiku: top 6 PCs of per-count means hold 95% of variance; helix-like curve with ringing",
            f"Qwen3-1.7B layer {q['best_layer']}: 6 PCs hold {qgeo['var'].sum() * 100:.0f}% (3 PCs {qgeo['var'][:3].sum() * 100:.0f}%); PC1–2 angle turns {qgeo['deg_per_char']:.2f}°/char",
            "6 PCs ≥ 80% of variance of the per-count means",
            qgeo["var"].sum() >= 0.8,
        ),
        verdict_row(
            "The count is represented in the very first layers",
            "Haiku: R² 0.985 already after layer 1 (T-Tech: early peak not replicated in open models)",
            f"Qwen3-1.7B: best R² within layers 0–2 = {summary['q_early_r2']:.3f} (best overall {qb['r2']:.3f} at layer {q['best_layer']})",
            "early-layer R² ≥ 0.9 × the best layer's R²",
            summary["q_early_r2"] >= 0.9 * qb["r2"],
        ),
        verdict_row(
            "The model breaks lines where the text does",
            "Haiku predicts newlines at the wrap width (newline probability 0.79 in their example)",
            f"AUC of P(newline) for true line breaks: Qwen3-1.7B {q['newline_auc']:.3f}, GPT-2 small {g['newline_auc']:.3f}",
            "Qwen3-1.7B AUC ≥ 0.9",
            q["newline_auc"] >= 0.9,
        ),
    ]
    write_entry(SLUG, {"qwen": q, "gpt2": g, "widths": WIDTHS}, summary, repl, models=["Qwen/Qwen3-1.7B", "gpt2"], runtime_s=time.time() - t0, ndigits=3)
    log.info("linebreak summary: %s", summary)
