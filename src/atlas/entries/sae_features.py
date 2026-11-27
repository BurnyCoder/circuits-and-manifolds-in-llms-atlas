"""Sparse-autoencoder features and feature steering ("Golden Gate GPT-2"), recomputed on GPT-2 small.

Background: Bricken et al. 2023 ("Towards Monosemanticity", https://transformer-circuits.pub/2023/monosemantic-features)
and Templeton et al. 2024 ("Scaling Monosemanticity", https://transformer-circuits.pub/2024/scaling-monosemanticity/):
SAEs decompose activations into sparse, often interpretable features; clamping the "Golden Gate Bridge" feature made
Claude 3 Sonnet talk about the bridge. Open SAEs: Bloom 2024 GPT-2 small residual SAEs (SAELens release
"gpt2-small-res-jb", 24,576 latents per layer).
Here (layer 8, resid_pre): pick features by their activation on short concept probes (Golden Gate Bridge, Python code,
French, DNA, Harry Potter), check specificity on unrelated text, collect top-activating contexts from WikiText-103,
compute each feature's top logit effects (W_dec · W_U), and steer generation by adding α·W_dec[f] at layer 8 (all
positions) with a norm-matched random direction as control. Topic adherence = share of completions containing the
concept's keywords. Neuronpedia dashboards for the same features are linked for comparison.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from ..common.models import hooked

log = get_logger()
SLUG = "sae-features"
LAYER = 8
HOOK = f"blocks.{LAYER}.hook_resid_pre"
CONCEPTS = {
    "Golden Gate Bridge": {"probe": "We drove across the Golden Gate Bridge into San Francisco", "focus": " Bridge", "keywords": ["golden gate", "bridge", "san francisco"]},
    "Python code": {"probe": "def main():\n    for i in range(10):\n        print(i)", "focus": " print", "keywords": ["def ", "print", "return", "import", "(", "=="]},
    "French": {"probe": "Je suis allé au marché avec ma mère et nous avons acheté du pain", "focus": " pain", "keywords": [" le ", " la ", " et ", " de ", " je ", " nous "]},
    "DNA": {"probe": "The DNA sequence was amplified by PCR and the genome was sequenced", "focus": " genome", "keywords": ["dna", "gene", "genome", "protein", "sequence", "cell"]},
    "Harry Potter": {"probe": "Harry Potter raised his wand while Hermione and Ron watched at Hogwarts", "focus": " Hogwarts", "keywords": ["harry", "hogwarts", "wizard", "magic", "wand", "potter"]},
}
NEUTRAL = ["My favourite thing to do on weekends is", "Yesterday I went to the store and", "The most important thing in life is"]


def compute(ctx) -> None:
    from datasets import load_dataset
    from sae_lens import SAE

    t0 = time.time()
    model = hooked("gpt2", device=ctx.device)
    loaded = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id=HOOK, device=ctx.device)
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    W_dec = sae.W_dec.detach()

    @torch.no_grad()
    def feats(text):
        toks = model.to_tokens(text)
        _, cache = model.run_with_cache(toks, names_filter=HOOK, stop_at_layer=LAYER + 1)
        return toks, sae.encode(cache[HOOK][0])  # [pos, n_feats]

    # Background activation: mean max-activation of every feature over unrelated text (specificity control).
    text = "\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"] if len(t) > 200)
    ids = model.to_tokens(text[: (200_000 if ctx.quick else 1_200_000)], prepend_bos=False)[0]
    chunks = ids[: (len(ids) // 128) * 128].view(-1, 128)
    chunks = torch.cat([torch.full((len(chunks), 1), model.tokenizer.bos_token_id, device=chunks.device), chunks], 1)
    bg_max = torch.zeros(W_dec.shape[0], device=ctx.device)
    store = []
    with torch.no_grad():
        for i in range(0, len(chunks), 32):
            c = chunks[i : i + 32]
            _, cache = model.run_with_cache(c, names_filter=HOOK, stop_at_layer=LAYER + 1)
            f = sae.encode(cache[HOOK])  # [b, 129, F]
            bg_max = torch.maximum(bg_max, f[:, 1:].amax((0, 1)))
            store.append((c.cpu(), f[:, 1:].max(1).values.cpu().half()))
    results = {}
    g = torch.Generator(device=ctx.device).manual_seed(ctx.seed)
    for name, spec in CONCEPTS.items():
        toks, f = feats(spec["probe"])
        strs = model.to_str_tokens(toks)
        pos = max(i for i, s in enumerate(strs) if s == spec["focus"]) if spec["focus"] in strs else len(strs) - 1
        act = f[pos]
        # Specificity score: activation on the concept token relative to the feature's typical maximum on WikiText.
        score = act / (bg_max.median() + 0.25 * bg_max + 1e-3)
        fid = int(score.argmax())
        # Top-activating WikiText contexts for this feature.
        best = []
        for c, fm in store:
            v = fm[:, fid].float()
            for j in v.topk(min(3, len(v))).indices.tolist():
                best.append((float(v[j]), c[j]))
        best.sort(key=lambda x: -x[0])
        contexts = []
        for val, c in best[:8]:
            with torch.no_grad():
                _, cache = model.run_with_cache(c[None].to(ctx.device), names_filter=HOOK, stop_at_layer=LAYER + 1)
                fa = sae.encode(cache[HOOK][0])[:, fid].float().cpu().numpy()
            peak = int(fa[1:].argmax()) + 1
            lo, hi = max(1, peak - 12), min(len(c), peak + 5)
            contexts.append({"tokens": [model.tokenizer.decode([t]) for t in c[lo:hi].tolist()], "acts": fa[lo:hi], "max": float(val)})
        logit_fx = (W_dec[fid] @ model.W_U).float()
        top_up = [model.tokenizer.decode([t]) for t in logit_fx.topk(10).indices.tolist()]
        top_down = [model.tokenizer.decode([t]) for t in (-logit_fx).topk(10).indices.tolist()]
        # Steering: add α·W_dec[f] at layer 8 (all positions); topic adherence by keywords; norm-matched random control.
        rnd = torch.randn(W_dec.shape[1], generator=g, device=ctx.device)
        rnd = rnd / rnd.norm() * W_dec[fid].norm()
        steer = []
        for alpha in ([0, 40] if ctx.quick else [0, 10, 20, 40, 80]):
            for kind, vec in (("feature", W_dec[fid]), ("random", rnd)):
                if alpha == 0 and kind == "random":
                    continue

                def hook(x, hook, vec=vec, alpha=alpha):
                    return x + alpha * vec

                comps = []
                for p in NEUTRAL:
                    for k in range(2 if ctx.quick else 4):
                        torch.manual_seed(ctx.seed + k)
                        with model.hooks(fwd_hooks=[(HOOK, hook)] if alpha else []):
                            o = model.generate(p, max_new_tokens=30, temperature=0.8, top_p=0.9, verbose=False, use_past_kv_cache=False)
                        comps.append(o[len(p) :])
                        log.info("SAE steer %s α=%s %s | %s ||| %s", name, alpha, kind, p, comps[-1].replace("\n", " "))
                rate = float(np.mean([any(kw in c.lower() for kw in spec["keywords"]) for c in comps]))
                steer.append({"alpha": alpha, "kind": kind, "on_topic": rate, "examples": comps[:3]})
        results[name] = {
            "feature": fid,
            "probe": spec["probe"],
            "probe_tokens": strs,
            "probe_acts": f[:, fid].float().cpu().numpy(),
            "contexts": contexts,
            "logit_up": top_up,
            "logit_down": top_down,
            "steer": steer,
            "neuronpedia": f"https://www.neuronpedia.org/gpt2-small/{LAYER}-res-jb/{fid}",
        }
        log.info("SAE feature for %s: #%d; up-logits %s; steering %s", name, fid, top_up[:5], [(s["alpha"], s["kind"], s["on_topic"]) for s in steer])
    gg = results["Golden Gate Bridge"]["steer"]
    on = lambda rows, a, k: next((r["on_topic"] for r in rows if r["alpha"] == a and r["kind"] == k), float("nan"))  # noqa: E731
    a_hi = 40
    lifts = [on(r["steer"], a_hi, "feature") - max(on(r["steer"], 0, "feature"), on(r["steer"], a_hi, "random")) for r in results.values()]
    summary = {
        "gg_feature": results["Golden Gate Bridge"]["feature"],
        "gg_base": on(gg, 0, "feature"),
        "gg_steer": on(gg, a_hi, "feature"),
        "gg_random": on(gg, a_hi, "random"),
        "alpha": a_hi,
        "n_concepts": len(results),
        "n_lift": int(sum(x >= 0.3 for x in lifts)),
        "median_lift": float(np.median(lifts)),
    }
    repl = [
        verdict_row(
            "Adding an SAE feature's direction steers generation toward its concept (Golden Gate)",
            "clamping the Golden Gate Bridge feature makes Claude 3 Sonnet obsessed with the bridge (Templeton et al. 2024)",
            f"GPT-2 small layer 8, feature #{summary['gg_feature']}: completions mentioning the bridge {summary['gg_base'] * 100:.0f}% → {summary['gg_steer'] * 100:.0f}% at α = {a_hi} (random direction {summary['gg_random'] * 100:.0f}%)",
            "≥ 50% on-topic and ≥ 30 points above baseline and random",
            summary["gg_steer"] >= 0.5 and summary["gg_steer"] - max(summary["gg_base"], summary["gg_random"]) >= 0.3,
            partial=summary["gg_steer"] - max(summary["gg_base"], summary["gg_random"]) >= 0.15,
        ),
        verdict_row(
            "Feature steering generalizes across concepts",
            "many features steer behaviour (feature steering section)",
            f"{summary['n_lift']}/{len(results)} concepts gain ≥ 30 points of on-topic completions at α = {a_hi} (median lift {summary['median_lift'] * 100:.0f} points)",
            "≥ 3 of 5 concepts",
            summary["n_lift"] >= 3,
            partial=summary["n_lift"] >= 2,
        ),
    ]
    write_entry(SLUG, {"concepts": results, "layer": LAYER}, summary, repl, models=["gpt2", "gpt2-small-res-jb SAE"], runtime_s=time.time() - t0, ndigits=3)
