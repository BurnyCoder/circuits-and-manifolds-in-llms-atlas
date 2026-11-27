"""Days of the week and months of the year live on circles (Engels et al. 2024), and the circle is causally used.

Recomputes Engels, Michaud, Liao, Gurnee, Tegmark, "Not All Language Model Features Are One-Dimensionally Linear"
(https://arxiv.org/abs/2405.14860, ICLR 2025):
- Geometry: residual-stream activations at day/month tokens, with the template's mean removed (so only the
  day/month identity remains), projected on their top principal components, arrange the 7 days / 12 months in their
  cyclic order (paper Fig. 1: GPT-2 small layer 7; Mistral 7B). We test cyclic order with a permutation null and
  compare against sets of random words (control).
- SAE view (GPT-2 small, layer 7, SAELens ``gpt2-small-res-jb`` — the Bloom 2024 SAEs the paper uses): reconstructing
  day-token activations from the SAE latents that fire most on days gives the circle; we also show how individual
  latents tile it (each prefers a few neighbouring days), the "capture vs. dilution" question of Bhalla et al. 2026.
- Causal test (Qwen3-1.7B, day arithmetic it solves, "If today is Monday, then two days later it is"): patching
  the 2-D circle component at the day token to another day's position shifts the model's answer by the same amount
  (paper §5 uses circular probes on Mistral/Llama); control: the same patch in a random 2-D subspace.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.geometry import cyclic_order_pvalue, cyclic_order_score, pca
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from ..common.models import hooked

log = get_logger()
SLUG = "calendar-circles"
DAYS = [" Monday", " Tuesday", " Wednesday", " Thursday", " Friday", " Saturday", " Sunday"]
MONTHS = [" January", " February", " March", " April", " May", " June", " July", " August", " September", " October", " November", " December"]
CONTROL_WORDS = " apple river chair window garden doctor bottle mountain pencil yellow castle engine ocean ticket violin basket rabbit summer silver forest".split()
TEMPLATES = [
    "I will see you on{X}.",
    "The meeting was moved to{X}.",
    "We usually go shopping on{X} afternoon.",
    "Her birthday party is on{X}.",
    "Everything changed that{X} evening.",
    "The report is due next{X}.",
    "He called me last{X} night.",
    "Let's plan the trip for{X}.",
    "The store is closed every{X}.",
    "It rained all day on{X}.",
    "My favourite time is{X} morning.",
    "They arrived early on{X}.",
    "The concert happens in{X}.",
    "School starts again in{X}.",
    "Prices usually rise in{X}.",
    "We first met in{X}.",
]
# Day-arithmetic prompts Qwen3-1.7B actually solves. CPU check 2026-11-27: "the day after" 7/7, "two days later" 6/7;
# Engels et al.'s "Let's do some day of the week math. N days from D is" scored 0-57% per N, so centroids built from
# the correct answer did not reflect what the model represents. Prefixes add paraphrase variety for averaging.
DAY_TASK = {1: "If today is{d}, then the day after is", 2: "If today is{d}, then two days later it is", 3: "If today is{d}, then three days later it is"}
DAY_PREFIXES = ["", "Quick calendar question. ", "Think about the days of the week. "]


def day_prompts() -> list[tuple[int, int, str]]:
    """(k, start day d, prompt) for every offset k in DAY_TASK, start day and prefix; the answer day is (d + k) % 7."""
    return [(k, d, pre + DAY_TASK[k].format(d=DAYS[d])) for pre in DAY_PREFIXES for k in DAY_TASK for d in range(7)]


def _positions(tok, texts: list[str], word: str) -> list[int]:
    """Index of the (single) token of ``word`` in each text (prefix tokenized separately; word has a leading space)."""
    out = []
    for t in texts:
        prefix = t[: t.index(word)]
        out.append(len(tok(prefix, add_special_tokens=False).input_ids))
    return out


def _single(tok, words: list[str]) -> bool:
    return all(len(tok(w, add_special_tokens=False).input_ids) == 1 for w in words)


@torch.no_grad()
def activations(model_fwd, tok, words: list[str], layer: int, device, bos: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Residual activation at each word token for every template: returns X [n, d], word labels, template labels."""
    X, y, g = [], [], []
    for ti, tpl in enumerate(TEMPLATES):
        texts = [tpl.replace("{X}", w) for w in words]
        pos = [_positions(tok, [t], w)[0] for t, w in zip(texts, words)]
        for t, w, p in zip(texts, words, pos):
            ids = tok(t, add_special_tokens=False, return_tensors="pt").input_ids
            if bos:
                ids = torch.cat([torch.tensor([[tok.bos_token_id]]), ids], 1)
                p += 1
            h = model_fwd(ids.to(device), layer)  # [1, pos, d]
            X.append(h[0, p].float().cpu().numpy())
            y.append(words.index(w))
            g.append(ti)
    return np.stack(X), np.array(y), np.array(g)


def circle_stats(X: np.ndarray, y: np.ndarray, g: np.ndarray, n_cls: int) -> dict:
    """Remove template means, PCA, class-mean angles in PC1–PC2, cyclic-order score + permutation p-value."""
    Xc = X.copy()
    for t in np.unique(g):
        Xc[g == t] -= Xc[g == t].mean(0)
    scores, comps, var, mu = pca(Xc, 3)
    means = np.stack([scores[y == c, :2].mean(0) for c in range(n_cls)])
    ang = np.arctan2(means[:, 1], means[:, 0])
    s = cyclic_order_score(ang)
    return {"points": scores, "labels": y, "means": means, "var": var, "score": s, "p": cyclic_order_pvalue(s, n_cls, 5000), "comps": comps, "mu": mu}


def gpt2_part(ctx) -> dict:
    model = hooked("gpt2", device=ctx.device)
    tok = model.tokenizer

    def fwd(ids, layer):
        _, cache = model.run_with_cache(ids, names_filter=f"blocks.{layer}.hook_resid_pre", stop_at_layer=layer + 1)
        return cache[f"blocks.{layer}.hook_resid_pre"]

    out = {}
    for name, words in (("days", DAYS), ("months", MONTHS)):
        assert _single(tok, words), name
        X, y, g = activations(fwd, tok, words, 7, ctx.device, bos=True)
        st = circle_stats(X, y, g, len(words))
        out[name] = st
        out[name + "_X"] = (X, y, g)
        log.info("GPT-2 L7 %s: cyclic order %.2f (p=%.4f), PC var %s", name, st["score"], st["p"], np.round(st["var"], 3))
    # Control: random word sets of the same sizes.
    rng = np.random.default_rng(ctx.seed)
    ctrl = []
    for n in (7, 12):
        for _ in range(5 if ctx.quick else 20):
            words = [" " + w for w in rng.choice(CONTROL_WORDS, n, replace=False)]
            X, y, g = activations(fwd, tok, words, 7, ctx.device, bos=True)
            st = circle_stats(X, y, g, n)
            ctrl.append({"n": n, "score": st["score"], "p": st["p"]})
    out["control"] = ctrl

    # SAE view: GPT-2 small layer-7 residual SAE (Bloom 2024, via SAELens).
    from sae_lens import SAE

    loaded = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.7.hook_resid_pre", device=ctx.device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    for name in ("days", "months"):
        X, y, g = out[name + "_X"]
        with torch.no_grad():
            f = sae.encode(torch.tensor(X, device=ctx.device)).cpu().numpy()  # [n, 24576]
        n_cls = len(np.unique(y))
        cls_mean = np.stack([f[y == c].mean(0) for c in range(n_cls)])  # [cls, feats]
        top = np.argsort(-cls_mean.max(0))[:24]  # latents most strongly active on some day/month
        top = top[cls_mean[:, top].max(0) > 0]
        Wd = sae.W_dec.detach().cpu().numpy()
        recon = f[:, top] @ Wd[top]
        st = circle_stats(recon, y, g, n_cls)
        out[name]["sae"] = {
            "features": top.tolist(),
            "tiling": cls_mean[:, top].T,  # [features, classes] mean activation
            "points": st["points"],
            "means": st["means"],
            "score": st["score"],
            "p": st["p"],
            "var": st["var"],
        }
        log.info("GPT-2 SAE %s: %d latents, cyclic order %.2f (p=%.4f)", name, len(top), st["score"], st["p"])
    return out


def qwen_part(ctx) -> dict:
    """Geometry layer sweep + causal circle patching on Qwen3-1.7B."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(ctx.device).eval()
    layers = model.model.layers
    assert _single(tok, DAYS) and _single(tok, MONTHS)

    def fwd(ids, layer):
        store = {}
        h = layers[layer].register_forward_hook(lambda m, i, o: store.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        model(ids)
        h.remove()
        return store["h"]

    sweep_layers = [2, 5, 8, 11, 14, 17, 20] if not ctx.quick else [5, 11]
    geo = {}
    for L in sweep_layers:
        for nm, words in (("days", DAYS), ("months", MONTHS)):
            X, y, g = activations(fwd, tok, words, L, ctx.device, bos=False)
            st = circle_stats(X, y, g, len(words))
            geo.setdefault(nm, []).append({"layer": L, "score": st["score"], "p": st["p"], "var": st["var"], "points": st["points"], "means": st["means"], "_st": st, "_X": (X, y, g)})
        log.info("Qwen3 L%d: days order %.2f, months order %.2f", L, geo["days"][-1]["score"], geo["months"][-1]["score"])

    # Causal test on day arithmetic the model solves ("If today is Monday, then two days later it is" -> answer day).
    day_ids = [tok(w, add_special_tokens=False).input_ids[0] for w in DAYS]
    prompts = day_prompts()

    @torch.no_grad()
    def answer(text, hook_layer=None, pos=None, delta=None):
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(ctx.device)
        hk = None
        if hook_layer is not None:
            def fn(m, i, o):
                hs = o[0] if isinstance(o, tuple) else o
                hs[:, pos] = hs[:, pos] + delta.to(hs.dtype)
                return o
            hk = layers[hook_layer].register_forward_hook(fn)
        logits = model(ids).logits[0, -1]
        if hk:
            hk.remove()
        return int(logits[day_ids].float().argmax().item())  # index of the predicted day among the 7

    clean_correct = [answer(t) == (d + k) % 7 for k, d, t in prompts]
    acc = float(np.mean(clean_correct))
    log.info("Qwen3 day-arithmetic accuracy: %.3f", acc)
    rng = np.random.default_rng(ctx.seed)
    causal = []
    for entry in geo["days"]:
        L = entry["layer"]
        st = entry["_st"]
        comps = st["comps"]  # top-3 PCs of template-demeaned day activations
        X, y, g = entry["_X"]
        # Class means of the *demeaned* activations in the 2-D circle plane, and in a random 2-D plane (control).
        Xc = X.copy()
        for t in np.unique(g):
            Xc[g == t] -= Xc[g == t].mean(0)
        P2 = comps[:2]
        R = np.linalg.qr(rng.normal(size=(X.shape[1], 2)))[0].T
        m_circle = np.stack([(Xc[y == c] @ P2.T).mean(0) for c in range(7)])
        m_rand = np.stack([(Xc[y == c] @ R.T).mean(0) for c in range(7)])
        succ, succ_rand = [], []
        for k, d, text in prompts:
            if not clean_correct[prompts.index((k, d, text))]:
                continue
            j = int(rng.integers(1, 7))  # shift the start day by j
            target = (d + j + k) % 7
            pos = len(tok(text[: text.index(DAYS[d])], add_special_tokens=False).input_ids)  # index of the start-day token
            for P, M, acc_list in ((P2, m_circle, succ), (R, m_rand, succ_rand)):
                delta = P.T @ (M[(d + j) % 7] - M[d])  # move along the plane from day d's mean to day d+j's mean
                acc_list.append(answer(text, L, pos, torch.tensor(delta, device=ctx.device)) == target)
        causal.append({"layer": L, "success": float(np.mean(succ)) if succ else 0.0, "success_random_plane": float(np.mean(succ_rand)) if succ_rand else 0.0, "n": len(succ)})
        log.info("Qwen3 L%d causal circle patch: %.3f (random plane %.3f, n=%d)", L, causal[-1]["success"], causal[-1]["success_random_plane"], len(succ))
    for nm in geo:
        for e in geo[nm]:
            e.pop("_st")
            e.pop("_X")
    del model
    torch.cuda.empty_cache()
    return {"geo": geo, "accuracy": acc, "causal": causal, "n_prompts": len(prompts)}


def compute(ctx) -> None:
    t0 = time.time()
    g2 = gpt2_part(ctx)
    qw = qwen_part(ctx)
    ctrl = g2["control"]
    ctrl_rate = float(np.mean([c["p"] < 0.05 for c in ctrl]))
    best_q = max(qw["causal"], key=lambda c: c["success"])
    best_days = max(qw["geo"]["days"], key=lambda e: e["score"])
    best_months = max(qw["geo"]["months"], key=lambda e: e["score"])
    summary = {
        "g2_days_score": g2["days"]["score"],
        "g2_days_p": g2["days"]["p"],
        "g2_months_score": g2["months"]["score"],
        "g2_months_p": g2["months"]["p"],
        "g2_sae_days_score": g2["days"]["sae"]["score"],
        "g2_sae_months_score": g2["months"]["sae"]["score"],
        "g2_sae_days_n": len(g2["days"]["sae"]["features"]),
        "ctrl_sig_rate": ctrl_rate,
        "n_ctrl": len(ctrl),
        "q_days_best_score": best_days["score"],
        "q_days_best_layer": best_days["layer"],
        "q_months_best_score": best_months["score"],
        "q_months_best_layer": best_months["layer"],
        "q_acc": qw["accuracy"],
        "q_causal_best": best_q["success"],
        "q_causal_best_layer": best_q["layer"],
        "q_causal_best_rand": best_q["success_random_plane"],
        "q_causal_n": best_q["n"],
    }
    repl = [
        verdict_row(
            "Days and months are arranged in cyclic order (GPT-2 small, layer 7)",
            "circles for days, months, years in GPT-2 L7 (Fig. 1)",
            f"days order {g2['days']['score']:.2f} (p = {g2['days']['p']:.3f}); months {g2['months']['score']:.2f} (p = {g2['months']['p']:.4f}); "
            f"random-word control significant in {ctrl_rate * 100:.0f}% of {len(ctrl)} sets",
            "both p < 0.05 and control false-positive rate ≤ 10%",
            g2["days"]["p"] < 0.05 and g2["months"]["p"] < 0.05 and ctrl_rate <= 0.10,
            partial=(g2["days"]["p"] < 0.05) != (g2["months"]["p"] < 0.05),
        ),
        verdict_row(
            "SAE latents reconstruct the circle",
            "SAE-cluster reconstruction shows the circle (Fig. 1)",
            f"reconstruction from the top {len(g2['days']['sae']['features'])} day latents: cyclic order {g2['days']['sae']['score']:.2f}; months {g2['months']['sae']['score']:.2f}",
            "cyclic order ≥ 0.7 for both",
            g2["days"]["sae"]["score"] >= 0.7 and g2["months"]["sae"]["score"] >= 0.7,
        ),
        verdict_row(
            "The circle is causally used for day arithmetic (Qwen3-1.7B)",
            "circular-probe patching shifts the answer in Mistral 7B / Llama 3 8B (§5)",
            f"accuracy {qw['accuracy'] * 100:.0f}% on {qw['n_prompts']} prompts; best layer {best_q['layer']}: patch success {best_q['success'] * 100:.0f}% "
            f"vs {best_q['success_random_plane'] * 100:.0f}% for a random 2-D plane (n = {best_q['n']})",
            "some layer reaches ≥ 50% success with its random-plane control ≤ 20%",
            best_q["success"] >= 0.5 and best_q["success_random_plane"] <= 0.2,
            partial=best_q["success"] > best_q["success_random_plane"] + 0.2,
        ),
    ]
    clean = lambda st: {k: st[k] for k in ("points", "labels", "means", "var", "score", "p", "sae") if k in st}  # noqa: E731
    payload = {
        "days": DAYS,
        "months": MONTHS,
        "gpt2": {"days": clean(g2["days"]), "months": clean(g2["months"]), "control": ctrl},
        "qwen": qw,
    }
    write_entry(SLUG, payload, summary, repl, models=["gpt2", "Qwen/Qwen3-1.7B"], runtime_s=time.time() - t0, ndigits=3)
    log.info("calendar summary: %s", summary)
