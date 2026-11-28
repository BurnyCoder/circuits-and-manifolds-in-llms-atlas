"""Meandering on manifolds: emotion trajectories while a model reads a story (Bigelow et al. 2026), on Qwen2.5-1.5B-Instruct.

Paper: Bigelow, Sarfati, Wurgaft, Lewis, McGrath, Merullo, Geiger, Lubana, "Stories in Space: In-Context Learning
Trajectories in Conceptual Belief Space" (https://arxiv.org/abs/2605.12412, CC BY 4.0; Goodfire post
https://www.goodfire.com/research/stories-in-space). Llama-3.1-8B-Instruct reads SimpleStories sentence by sentence;
after each prefix it is asked to rate six emotions 0–10 (behavioural readout = expected rating from the digit
distribution); ridge probes from layer-9 activations predict the ratings (RMSE 0.09 for emotions), behaviour and
activation geometry agree (distance-matrix r = .92), and the emotion layout matches human valence–arousal (r = .93).
Here: Qwen2.5-1.5B-Instruct, SimpleStories, six emotions (happy, sad, angry, surprised, afraid, disgusted), per-sentence
ratings on a 0–9 scale (single-digit answers: Qwen tokenizes "10" as "1","0"), ridge probes per layer (held-out stories), and 2-D trajectories in a valence–arousal plane built from human
norms (Warriner et al. 2013).
"""

from __future__ import annotations

import re
import time

import numpy as np
import torch

from ..common.geometry import ridge_r2, validated_probe
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from ..common.stats import ci_rule
from .emotions import norms

log = get_logger()
SLUG = "stories-in-space"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMO = ["happy", "sad", "angry", "surprised", "afraid", "disgusted"]
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
N_FRESH = 200  # confirmatory stories after the main run's (reports/planned-runs.md, review follow-ups 2026-11-28)


def rsa(A: np.ndarray, Y: np.ndarray, sid: np.ndarray, rng, n_null: int = 100) -> tuple[float, float, int]:
    """Behaviour vs activation geometry, representational-similarity style (Kriegeskorte et al. 2008,
    https://doi.org/10.3389/neuro.06.004.2008): Spearman correlation between pairwise distances of the prefixes' rating
    vectors and of their activations (standardized, top-50 PCs). Null: the same with whole stories' rating blocks
    shuffled between stories of equal length (keeps within-story dependence). Returns (r, null 95th pct, n prefixes)."""
    from scipy.stats import spearmanr

    A = (A - A.mean(0)) / (A.std(0) + 1e-6)
    A = A @ np.linalg.svd(A, full_matrices=False)[2][:50].T
    sub = rng.permutation(len(Y))[:600]  # at most 600 prefixes (~180k pairs)
    iu = np.triu_indices(len(sub), 1)

    def dists(M):
        return np.linalg.norm(M[sub][:, None] - M[sub][None], axis=-1)[iu]

    da = dists(A)
    r_geom = float(spearmanr(dists(Y), da).statistic)
    lengths = {si: (sid == si).sum() for si in np.unique(sid)}
    null = []
    for _ in range(n_null):
        Yp = Y.copy()
        for L_ in set(lengths.values()):  # shuffle rating blocks among stories with the same number of sentences
            group = [si for si, n_ in lengths.items() if n_ == L_]
            for src, dst in zip(group, rng.permutation(group)):
                Yp[sid == dst] = Y[sid == src]
        null.append(float(spearmanr(dists(Yp), da).statistic))
    return r_geom, float(np.quantile(null, 0.95)), len(sub)


def bow_r2(texts: list[str], Y: np.ndarray, sid: np.ndarray) -> float:
    """Bag-of-words baseline (reported, no verdict): TF-IDF unigrams + bigrams of the prefix text, ridge with alpha
    chosen on the same validation stories as the activation probe, R² on the test stories."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    from ..common.geometry import r2_score

    test = sid % 5 == 0
    val = np.isin(sid, np.unique(sid[~test])[::5])
    fit = ~test & ~val
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit([t for t, f in zip(texts, ~test) if f])
    T = vec.transform(texts)
    a = max([1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], key=lambda a: r2_score(Y[val], Ridge(alpha=a).fit(T[fit], Y[fit]).predict(T[val])))
    return r2_score(Y[test], Ridge(alpha=a).fit(T[~test], Y[~test]).predict(T[test]))


def compute(ctx) -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(ctx.device).eval()
    ds = load_dataset("SimpleStories/SimpleStories", split="test")
    rng = np.random.default_rng(ctx.seed)
    n_st = 20 if ctx.quick else 160
    stories = []
    n_fresh = 20 if ctx.quick else N_FRESH
    for i in rng.permutation(len(ds)):
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ds[int(i)]["story"]) if s.strip()]
        if 6 <= len(sents) <= 14:
            stories.append(sents)
        if len(stories) >= n_st + n_fresh:
            break
    stories, fresh = stories[:n_st], stories[n_st:]  # the main run's stories are exactly those it always used
    # Ratings on a 0–9 scale: every answer is one token. (Qwen splits "10" into "1","0", so a 0–10 scale would fold
    # P(10) into P(1) and skew the expected rating.)
    digits = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(10)]
    assert len(set(digits)) == 10
    nl = model.config.num_hidden_layers
    layers = list(range(2, nl, 3))
    tok.padding_side = "left"

    @torch.no_grad()
    def forward(texts, hidden=False):
        """Batched forward with left padding; position ids from the attention mask so padding does not shift positions."""
        b = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(ctx.device)
        pos = (b.attention_mask.cumsum(-1) - 1).clamp(min=0)
        return model(input_ids=b.input_ids, attention_mask=b.attention_mask, position_ids=pos, output_hidden_states=hidden)

    def collect(stories):
        """Last-token activations per layer, expected 0–9 ratings per emotion, story ids and prefix texts."""
        X = {L: [] for L in layers}
        Y, sid, texts = [], [], []
        for si, sents in enumerate(stories):
            prefixes = [" ".join(sents[:t]) for t in range(1, len(sents) + 1)]
            hs = forward(prefixes, hidden=True).hidden_states  # all prefixes of the story in one batch
            for L in layers:
                X[L] += list(hs[L][:, -1].float().cpu().numpy())
            qs = [
                tok.apply_chat_template([{"role": "user", "content": f"Story so far: \"{pre}\"\n\nOn a scale from 0 to 9, how {e} does the story feel so far? Answer with a single digit."}], add_generation_prompt=True, tokenize=False)
                for pre in prefixes
                for e in EMO
            ]
            p = forward(qs).logits[:, -1, digits].float().softmax(-1).cpu().numpy().reshape(len(prefixes), len(EMO), 10)
            Y += list(p @ np.arange(10) / 9)  # expected rating in [0, 1] per prefix and emotion
            sid += [si] * len(prefixes)
            texts += prefixes
            log.info("story %d/%d (%d sentences): final-prefix ratings %s | %s", si + 1, len(stories), len(sents),
                     {e: round(float(r), 2) for e, r in zip(EMO, Y[-1])}, prefixes[-1])
        return {L: np.stack(v) for L, v in X.items()}, np.stack(Y), np.array(sid), texts

    X, Y, sid, _ = collect(stories)
    test = sid % 5 == 0
    probe = []
    for L in layers:
        A = X[L]
        # Ridge strength chosen on held-out training stories (grouped by story id), never on the test stories.
        r2 = ridge_r2(A[~test], Y[~test], A[test], Y[test], alpha=ALPHAS, groups=sid[~test])
        probe.append({"layer": L, "r2": r2})
        log.info("stories probe L%d: R² %.3f", L, r2)
    best = max(probe, key=lambda p: p["r2"])
    r_geom, r_null95, n_rsa = rsa(X[best["layer"]], Y, sid, rng)
    log.info("stories RSA: Spearman r = %.3f (story-shuffle null 95th pct %.3f) over %d prefixes", r_geom, r_null95, n_rsa)
    # Pre-stated confirmatory run on fresh stories: validation-chosen layer, bootstrap CI over test stories, BoW baseline,
    # RSA at the chosen layer.
    Xf, Yf, sidf, textsf = collect(fresh)
    cf = validated_probe(Xf, Yf, sidf, ALPHAS, ctx.seed)
    r_bow = bow_r2(textsf, Yf, sidf)
    rf, rf_null95, nf_rsa = rsa(Xf[cf["layer"]], Yf, sidf, rng)
    log.info("stories confirmatory (fresh): layer %d R² %.3f [%.3f, %.3f]; bag-of-words R² %.3f; RSA r %.3f (null 95th %.3f)",
             cf["layer"], cf["r2"], cf["r2_lo"], cf["r2_hi"], r_bow, rf, rf_null95)
    N = norms()
    va = np.array([N[e] for e in EMO])  # human valence/arousal of the six emotion words
    traj = []
    for si in range(min(12, len(stories))):
        w = Y[sid == si]
        w = w / w.sum(1, keepdims=True)
        traj.append({"sentences": stories[si], "ratings": Y[sid == si], "xy": w @ va})
    summary = {
        "n_stories": len(stories),
        "n_prefixes": int(len(Y)),
        "best_layer": best["layer"],
        "best_r2": best["r2"],
        "r_geom": r_geom,
        "r_geom_null95": r_null95,
        "n_rsa_prefixes": n_rsa,
        "f_n_stories": len(fresh),
        "f_layer": cf["layer"],
        "f_r2": cf["r2"],
        "f_r2_lo": cf["r2_lo"],
        "f_r2_hi": cf["r2_hi"],
        "f_bow_r2": r_bow,
        "f_rsa": rf,
        "f_rsa_null95": rf_null95,
    }
    p1, part1 = ci_rule(cf["r2_lo"], cf["r2_hi"], 0.5)
    repl = [
        verdict_row(
            "The model's running emotional reading is linearly decodable",
            "ridge probes predict per-sentence emotion ratings (RMSE 0.09) from layer-9 activations (Llama-3.1-8B-Instruct)",
            f"held-out-story R² {best['r2']:.2f} at layer {best['layer']} ({len(Y)} prefixes from {len(stories)} stories)",
            "R² ≥ 0.5",
            best["r2"] >= 0.5,
            partial=best["r2"] >= 0.3,
        ),
        verdict_row(
            "Behaviour and activation geometry agree",
            "distance-matrix correlation r = .92 (emotions)",
            f"RSA Spearman r = {r_geom:.2f} over {n_rsa} prefixes (story-shuffle null 95th percentile {r_null95:.2f})",
            "r ≥ 0.6 (amendment 2026-11-27: computed over prefixes, not 3–6 emotion centroids)",
            r_geom >= 0.6,
            partial=r_geom >= 0.3,
        ),
        verdict_row(
            "Confirmatory: the running emotional reading is linearly decodable (fresh stories)",
            "ridge probes predict per-sentence emotion ratings (RMSE 0.09) from layer-9 activations (Llama-3.1-8B-Instruct)",
            f"{len(fresh)} fresh stories ({cf['n_test_groups']} test): R² {cf['r2']:.2f} [{cf['r2_lo']:.2f}, {cf['r2_hi']:.2f}] at validation-chosen layer {cf['layer']}; "
            f"bag-of-words baseline on the prefix text {r_bow:.2f}",
            "CI rule at 0.5 (pre-stated 2026-11-28)",
            p1,
            partial=part1,
        ),
        verdict_row(
            "Confirmatory: behaviour and activation geometry agree (fresh stories)",
            "distance-matrix correlation r = .92 (emotions)",
            f"RSA Spearman r = {rf:.2f} at layer {cf['layer']} over {nf_rsa} prefixes (story-shuffle null 95th percentile {rf_null95:.2f})",
            "r ≥ 0.6 and above the null's 95th percentile; partial if r ≥ 0.3 and above it (pre-stated 2026-11-28)",
            rf >= 0.6 and rf > rf_null95,
            partial=rf >= 0.3 and rf > rf_null95,
        ),
    ]
    write_entry(SLUG, {"emotions": EMO, "va": va, "probe": probe, "trajectories": traj}, summary, repl, models=[MODEL, "SimpleStories"], runtime_s=time.time() - t0, ndigits=3)
    del model
    torch.cuda.empty_cache()
