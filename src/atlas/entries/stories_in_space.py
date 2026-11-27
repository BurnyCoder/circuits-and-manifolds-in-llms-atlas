"""Meandering on manifolds: emotion trajectories while a model reads a story (Bigelow et al. 2026), on Qwen2.5-1.5B-Instruct.

Paper: Bigelow, Sarfati, Wurgaft, Lewis, McGrath, Merullo, Geiger, Lubana, "Stories in Space: In-Context Learning
Trajectories in Conceptual Belief Space" (https://arxiv.org/abs/2605.12412, CC BY 4.0; Goodfire post
https://www.goodfire.com/research/stories-in-space). Llama-3.1-8B-Instruct reads SimpleStories sentence by sentence;
after each prefix it is asked to rate six emotions 0–10 (behavioural readout = expected rating from the digit
distribution); ridge probes from layer-9 activations predict the ratings (RMSE 0.09 for emotions), behaviour and
activation geometry agree (distance-matrix r = .92), and the emotion layout matches human valence–arousal (r = .93).
Here: Qwen2.5-1.5B-Instruct, SimpleStories, six emotions (happy, sad, angry, surprised, afraid, disgusted), per-sentence
ratings, ridge probes per layer (held-out stories), and 2-D trajectories in a valence–arousal plane built from human
norms (Warriner et al. 2013).
"""

from __future__ import annotations

import re
import time

import numpy as np
import torch

from ..common.geometry import ridge_r2
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from .emotions import norms

log = get_logger()
SLUG = "stories-in-space"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMO = ["happy", "sad", "angry", "surprised", "afraid", "disgusted"]


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
    for i in rng.permutation(len(ds)):
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ds[int(i)]["story"]) if s.strip()]
        if 6 <= len(sents) <= 14:
            stories.append(sents)
        if len(stories) >= n_st:
            break
    digits = [tok(str(d), add_special_tokens=False).input_ids[0] for d in range(11)]
    nl = model.config.num_hidden_layers
    layers = list(range(2, nl, 3))
    X = {L: [] for L in layers}
    Y, sid, pos_in_story = [], [], []
    for si, sents in enumerate(stories):
        for t in range(1, len(sents) + 1):
            prefix = " ".join(sents[:t])
            ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
            with torch.no_grad():
                hs = model(ids, output_hidden_states=True).hidden_states
            for L in layers:
                X[L].append(hs[L][0, -1].float().cpu().numpy())
            qs = [tok.apply_chat_template([{"role": "user", "content": f"Story so far: \"{prefix}\"\n\nOn a scale from 0 to 10, how {e} does the story feel so far? Answer with a single number."}], add_generation_prompt=True, tokenize=False) for e in EMO]
            tok.padding_side = "left"
            b = tok(qs, return_tensors="pt", padding=True, add_special_tokens=False).to(ctx.device)
            with torch.no_grad():
                p = model(**b).logits[:, -1, digits].float().softmax(-1).cpu().numpy()  # [6, 11]
            Y.append(p @ np.arange(11) / 10)  # expected rating in [0, 1]
            sid.append(si)
            pos_in_story.append(t / len(sents))
        if si % 20 == 0:
            log.info("stories: %d/%d done", si + 1, len(stories))
    Y, sid = np.stack(Y), np.array(sid)
    test = sid % 5 == 0
    probe = []
    for L in layers:
        A = np.stack(X[L])
        r2 = ridge_r2(A[~test], Y[~test], A[test], Y[test], alpha=100.0)
        probe.append({"layer": L, "r2": r2})
        log.info("stories probe L%d: R² %.3f", L, r2)
    best = max(probe, key=lambda p: p["r2"])
    # Behaviour vs activation geometry: distances between the six emotions' "signature" prefixes.
    A = np.stack(X[best["layer"]])
    top = Y.argmax(1)
    beh_c = np.stack([Y[top == c].mean(0) if (top == c).any() else np.zeros(6) for c in range(6)])
    act_c = np.stack([A[top == c].mean(0) if (top == c).any() else np.zeros(A.shape[1]) for c in range(6)])
    present = [c for c in range(6) if (top == c).sum() >= 3]
    iu = np.triu_indices(len(present), 1)
    db = np.linalg.norm(beh_c[present][:, None] - beh_c[present][None], axis=-1)[iu]
    da = np.linalg.norm(act_c[present][:, None] - act_c[present][None], axis=-1)[iu]
    r_geom = float(np.corrcoef(db, da)[0, 1]) if len(db) > 2 else float("nan")
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
        "n_emotions_present": len(present),
    }
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
            f"r = {r_geom:.2f} over {len(present)} emotions with ≥ 3 dominant prefixes",
            "r ≥ 0.6",
            r_geom >= 0.6,
            partial=r_geom >= 0.3,
        ),
    ]
    write_entry(SLUG, {"emotions": EMO, "va": va, "probe": probe, "trajectories": traj}, summary, repl, models=[MODEL, "SimpleStories"], runtime_s=time.time() - t0, ndigits=3)
    del model
    torch.cuda.empty_cache()
