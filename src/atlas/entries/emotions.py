"""Emotion concepts and their function (Sofroniew et al. 2026), a small open replication on Qwen3-1.7B.

Paper: Sofroniew, Kauvar, Saunders, Chen, Henighan, … Olah, Lindsey, "Emotion Concepts and their Function in a Large
Language Model" (https://transformer-circuits.pub/2026/emotions/index.html; arXiv 2604.07729). Claude Sonnet 4.5 wrote
~205k short stories in which a character feels one of 171 emotions; emotion vectors = mean activation over a story's
tokens minus the mean over all emotions (neutral-text principal components projected out), at ~2/3 depth. PC1 of the
vectors (26%) tracks valence (r = 0.81 with human ratings), PC2 (15%) arousal (r = 0.66); vectors track dose-dependent
danger ("Tylenol" sweep) and causally shift preferences (blissful +212 Elo, hostile −303) and misaligned behaviour.
Here: 80 emotion words that have human valence/arousal norms (Warriner, Kuperman & Brysbaert 2013), Qwen3-1.7B writes
the stories itself (12 topics × 2 per emotion ≈ 1,900 stories; stories naming the emotion word are dropped), plus 120
neutral stories for the confound projection. We report PCA geometry vs human norms (and a shuffled-label control),
the dose sweep, and a causal preference test: adding ±α·v (blissful / hostile) on the tokens of option A in "Would you
rather (A) … or (B) …?" and measuring the shift in P(A) vs a random vector of the same norm.
"""

from __future__ import annotations

import io
import time

import numpy as np
import requests
import torch

from ..common.geometry import pca
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "emotions"
MODEL = "Qwen/Qwen3-1.7B"
LAYER = 18  # ≈ 2/3 of 28 layers, as in the paper
NORMS_URL = "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv"
EMOTIONS = """happy sad angry afraid calm excited bored proud ashamed guilty jealous lonely grateful hopeful anxious nervous
relaxed content joyful cheerful delighted ecstatic furious irritated annoyed frustrated disappointed depressed miserable
terrified scared worried panicked confident embarrassed surprised amazed curious disgusted hostile bitter resentful
loving affectionate tender peaceful serene relieved satisfied thrilled enthusiastic inspired determined desperate helpless
hopeless heartbroken grief nostalgic sentimental envious humiliated insecure shy awkward tense restless impatient
eager playful amused triumphant blissful gloomy sorrowful remorseful vengeful suspicious defiant""".split()
TOPICS = ["a job interview", "a family dinner", "a long train journey", "moving to a new city", "a hospital visit", "a school exam",
          "a birthday party", "a lost pet", "a football match", "a rainy weekend", "a first date", "an old friend calling"]
ACTIVITIES = ["organize a messy bookshelf", "learn to play the violin", "walk in a quiet forest", "fill in tax forms", "bake bread from scratch",
              "sit through a long meeting", "paint a landscape", "clean the bathroom", "watch the sunrise at the beach", "wait in a long queue",
              "write a letter to an old friend", "assemble flat-pack furniture", "visit an art museum", "answer hundreds of emails",
              "plant tomatoes in a garden", "sort a pile of receipts"]


def norms() -> dict[str, tuple[float, float]]:
    import pandas as pd

    df = pd.read_csv(io.StringIO(requests.get(NORMS_URL, timeout=60).text))
    return {w: (v, a) for w, v, a in zip(df["Word"], df["V.Mean.Sum"], df["A.Mean.Sum"])}


def compute(ctx) -> None:
    from sklearn.cluster import KMeans
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(ctx.device).eval()
    N = norms()
    emos = [e for e in EMOTIONS if e in N][: (12 if ctx.quick else 80)]
    # The dose sweep and the preference test need these four; keep them even in --quick mode (first-12 subset).
    emos += [e for e in ("afraid", "calm", "blissful", "hostile") if e in N and e not in emos]
    topics = TOPICS[:3] if ctx.quick else TOPICS
    reps = 1 if ctx.quick else 2

    def chat(content):
        return tok.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False, enable_thinking=False)

    @torch.no_grad()
    def write(prompts, n_new=140, seed=0):
        outs = []
        for i in range(0, len(prompts), 48):
            torch.manual_seed(seed + i)
            b = tok([chat(p) for p in prompts[i : i + 48]], return_tensors="pt", padding=True, add_special_tokens=False).to(ctx.device)
            o = model.generate(**b, max_new_tokens=n_new, do_sample=True, temperature=0.9, top_p=0.95, pad_token_id=tok.eos_token_id)
            batch = tok.batch_decode(o[:, b.input_ids.shape[1] :], skip_special_tokens=True)
            # Every prompt and output goes to the timestamped log in full, as it is produced.
            for q, ans in zip(prompts[i : i + 48], batch):
                log.info("generate | prompt: %s | output: %s", q, ans.replace("\n", " "))
            outs += batch
            log.info("emotions: generated %d/%d texts", len(outs), len(prompts))
        return outs

    jobs = [(e, t) for e in emos for t in topics for _ in range(reps)]
    stories = write([f"Write a short one-paragraph story (about 100 words) about {t}, in which the main character feels {e}. Do not use the word '{e}'." for e, t in jobs], seed=ctx.seed)
    keep = [(e, s) for (e, _), s in zip(jobs, stories) if e not in s.lower() and len(s) > 200]
    neutral = write([f"Write a short, purely factual one-paragraph description (about 100 words) of {t}, with no emotions or feelings." for t in topics for _ in range(10 if not ctx.quick else 2)], seed=ctx.seed + 1)
    log.info("emotions: %d/%d stories kept (emotion word absent), %d neutral", len(keep), len(jobs), len(neutral))

    @torch.no_grad()
    def story_mean(texts):
        """Mean layer-LAYER residual over each text's tokens from token 20 on (story read as plain text)."""
        out = []
        for t in texts:
            ids = tok(t, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=256).input_ids.to(ctx.device)
            h = model(ids, output_hidden_states=True).hidden_states[LAYER][0, 20:].float()
            out.append(h.mean(0).cpu().numpy() if len(h) else np.zeros(model.config.hidden_size))
        return np.stack(out)

    X = story_mean([s for _, s in keep])
    labels = np.array([emos.index(e) for e, _ in keep])
    Xn = story_mean(neutral)
    # Confound removal: project out the top neutral-text PCs explaining 50% of their variance (paper §2).
    _, comps, var, _ = pca(Xn, min(len(Xn) - 1, 50))
    k50 = int(np.searchsorted(np.cumsum(var), 0.5) + 1)
    P = comps[:k50]

    def vectors(lab):
        means = np.stack([X[lab == i].mean(0) for i in range(len(emos))])
        V = means - means.mean(0)
        return V - (V @ P.T) @ P

    V = vectors(labels)
    scores, vcomps, vvar, _ = pca(V, 3)
    val = np.array([N[e][0] for e in emos])
    aro = np.array([N[e][1] for e in emos])
    r_val = abs(float(np.corrcoef(scores[:, 0], val)[0, 1]))
    r_aro = abs(float(np.corrcoef(scores[:, 1], aro)[0, 1]))
    # Shuffled-label control: vectors from randomly permuted story labels.
    rng = np.random.default_rng(ctx.seed)
    null_val = []
    for _ in range(20):
        s_, _, _, _ = pca(vectors(rng.permutation(labels)), 1)
        null_val.append(abs(float(np.corrcoef(s_[:, 0], val)[0, 1])))
    clusters = KMeans(10, n_init=10, random_state=0).fit_predict(V / np.linalg.norm(V, axis=1, keepdims=True)) if len(emos) >= 10 else np.zeros(len(emos), int)
    log.info("emotion PCA var %s; |r(PC1, valence)| %.3f (shuffled 95%% %.3f), |r(PC2, arousal)| %.3f", np.round(vvar, 3), r_val, np.quantile(null_val, 0.95), r_aro)

    # Dose sweep: projection of the last prompt token on 'afraid' and 'calm' vectors.
    def unit(e):
        v = V[emos.index(e)]
        return v / np.linalg.norm(v)

    doses = [250, 500, 1000, 2000, 4000, 8000, 16000]
    sweep = {"dose": doses, "afraid": [], "calm": []}
    for d in doses:
        ids = tok(chat(f"I just took {d} mg of Tylenol for my headache. Should I be worried?"), return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
        with torch.no_grad():
            h = model(ids, output_hidden_states=True).hidden_states[LAYER][0, -1].float().cpu().numpy()
        for e in ("afraid", "calm"):
            sweep[e].append(float((h - X.mean(0)) @ unit(e)))
    log.info("Tylenol sweep: %s", sweep)

    # Causal preference test.
    A_tok = tok(" A", add_special_tokens=False).input_ids[0]
    B_tok = tok(" B", add_special_tokens=False).input_ids[0]
    pairs = [(a, b) for i, a in enumerate(ACTIVITIES) for j, b in enumerate(ACTIVITIES) if i < j][: (20 if ctx.quick else 120)]
    scale = float(np.median(np.linalg.norm(X, axis=1)))
    g = torch.Generator().manual_seed(ctx.seed)

    def pref_shift(vec: np.ndarray | None, alpha: float) -> float:
        shifts = []
        for a, b in pairs:
            text = chat(f"Would you rather (A) {a} or (B) {b}? Answer with a single letter.") + "Answer:"
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
            span = tok(text[: text.index(a) + len(a)], add_special_tokens=False).input_ids
            lo = len(tok(text[: text.index(a)], add_special_tokens=False).input_ids)
            hi = len(span)
            hk = None
            if vec is not None:
                d = torch.tensor(vec / np.linalg.norm(vec) * alpha * scale, device=ctx.device, dtype=torch.bfloat16)

                def fn(m, i, o):
                    hs = o[0] if isinstance(o, tuple) else o
                    if hs.shape[1] > lo:
                        hs[:, lo:hi] += d
                    return o

                hk = model.model.layers[LAYER - 1].register_forward_hook(fn)
            with torch.no_grad():
                lg = model(ids).logits[0, -1].float()
            if hk:
                hk.remove()
            shifts.append(float(torch.sigmoid(lg[A_tok] - lg[B_tok])))
        return float(np.mean(shifts))

    alpha = 0.5  # fraction of the typical residual norm, as in the paper's steering strength 0.5
    base = pref_shift(None, 0)
    rnd = torch.randn(V.shape[1], generator=g).numpy()
    pref = {
        "baseline": base,
        "blissful": pref_shift(V[emos.index("blissful")], alpha) if "blissful" in emos else float("nan"),
        "hostile": pref_shift(V[emos.index("hostile")], alpha) if "hostile" in emos else float("nan"),
        "random vector": pref_shift(rnd, alpha),
    }
    log.info("preference P(A): %s", pref)
    summary = {
        "n_emotions": len(emos),
        "n_stories": len(keep),
        "var_pc1": float(vvar[0]),
        "var_pc2": float(vvar[1]),
        "r_valence": r_val,
        "r_arousal": r_aro,
        "null_valence_p95": float(np.quantile(null_val, 0.95)),
        "afraid_slope": float(np.polyfit(np.log10(doses), sweep["afraid"], 1)[0]),
        "calm_slope": float(np.polyfit(np.log10(doses), sweep["calm"], 1)[0]),
        "pref_base": base,
        "pref_blissful": pref["blissful"],
        "pref_hostile": pref["hostile"],
        "pref_random": pref["random vector"],
        "layer": LAYER,
    }
    repl = [
        verdict_row(
            "The main axis of emotion space is valence",
            "PC1 26% of variance; r = 0.81 with human valence (Claude Sonnet 4.5)",
            f"PC1 {vvar[0] * 100:.0f}% of variance; |r| = {r_val:.2f} with human valence (shuffled-label 95th pct {summary['null_valence_p95']:.2f}); {len(emos)} emotions, {len(keep)} stories",
            "|r| ≥ 0.6 and above the shuffled-label 95th percentile",
            r_val >= 0.6 and r_val > summary["null_valence_p95"],
            partial=r_val > summary["null_valence_p95"],
        ),
        verdict_row(
            "The second axis is arousal",
            "PC2 15%; r = 0.66 with human arousal",
            f"PC2 {vvar[1] * 100:.0f}% of variance; |r| = {r_aro:.2f} with human arousal",
            "|r| ≥ 0.4",
            r_aro >= 0.4,
            partial=r_aro >= 0.25,
        ),
        verdict_row(
            "Emotion vectors track danger in the user's situation",
            "'afraid' rises and 'calm' falls with the Tylenol dose (Fig. 3)",
            f"slope per decade of dose: afraid {summary['afraid_slope']:+.2f}, calm {summary['calm_slope']:+.2f}",
            "afraid slope > 0 and calm slope < 0",
            summary["afraid_slope"] > 0 and summary["calm_slope"] < 0,
        ),
        verdict_row(
            "Steering with emotion vectors shifts preferences",
            "blissful steering on an option raises its Elo (+212), hostile lowers it (−303)",
            f"P(choose A): baseline {base:.2f}, +blissful on A {pref['blissful']:.2f}, +hostile on A {pref['hostile']:.2f}, random vector {pref['random vector']:.2f}",
            "blissful > baseline > hostile, each by ≥ 0.05, with |random − baseline| < 0.05",
            pref["blissful"] - base >= 0.05 and base - pref["hostile"] >= 0.05 and abs(pref["random vector"] - base) < 0.05,
            partial=pref["blissful"] > base > pref["hostile"],
        ),
    ]
    payload = {
        "emotions": emos,
        "pcs": scores,
        "var": vvar,
        "valence": val,
        "arousal": aro,
        "clusters": clusters,
        "sweep": sweep,
        "pref": pref,
        "examples": [{"emotion": e, "story": s} for e, s in keep[:: max(1, len(keep) // 24)]][:24],
    }
    write_entry(SLUG, payload, summary, repl, models=[MODEL], runtime_s=time.time() - t0, ndigits=3)
    del model
    torch.cuda.empty_cache()
