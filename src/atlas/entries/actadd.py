"""Activation addition (ActAdd): steering GPT-2-XL with the difference of two prompts' activations.

Paper: Turner, Thiergart, Leech, Udell, Vazquez, Mini, MacDiarmid, "Steering Language Models With Activation
Engineering" (https://arxiv.org/abs/2308.10248, CC BY 4.0). A steering vector is the difference of residual-stream
activations of a contrast pair (e.g. " Love" − " Hate", or "I talk about weddings constantly" − "I do not talk about
weddings constantly"), added with coefficient c at one layer to the first positions of the user's prompt (the paper
pads the pair to equal length). Paper: wedding vector → > 90% of completions on-topic vs ~2% unsteered; middle layers
(~6–24) work best; c usually < 15.
We sweep coefficients for two vectors on neutral/negative prompts, score completions (wedding words; sentiment with
distilbert-sst2, Apache-2.0), and compare against a random vector of the same norm (control). All completions are
logged in full and shown on the site.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from ..common.models import hooked

log = get_logger()
SLUG = "actadd"
WEDDING_WORDS = ["wedding", "bride", "groom", "marry", "married", "marriage", "ceremony", "honeymoon", "engaged", "vows"]
VECTORS = {
    "weddings": {"plus": "I talk about weddings constantly", "minus": "I do not talk about weddings constantly", "layer": 20, "prompts": ["I went up to my friend and said", "Did you hear the news? It turns out that", "The best thing about this weekend was"]},
    "love vs hate": {"plus": " Love", "minus": " Hate", "layer": 6, "prompts": ["I hate you because", "Honestly, my neighbours are", "The movie last night was"]},
}
COEFS = [0, 1, 2, 4, 8]


def steering_vector(model, plus: str, minus: str, layer: int) -> torch.Tensor:
    """[n_pos, d] difference of resid_pre activations of the padded contrast pair (ActAdd §3)."""
    tp, tm = model.to_tokens(plus), model.to_tokens(minus)
    n = max(tp.shape[1], tm.shape[1])
    pad = model.tokenizer.encode(" ")[0]  # the paper pads with spaces
    tp = torch.cat([tp, torch.full((1, n - tp.shape[1]), pad, device=tp.device)], 1)
    tm = torch.cat([tm, torch.full((1, n - tm.shape[1]), pad, device=tm.device)], 1)
    name = f"blocks.{layer}.hook_resid_pre"
    _, cp = model.run_with_cache(tp, names_filter=name)
    _, cm = model.run_with_cache(tm, names_filter=name)
    return (cp[name] - cm[name])[0]


@torch.no_grad()
def generate(model, prompt: str, vec: torch.Tensor | None, coef: float, layer: int, n: int, seed: int) -> list[str]:
    """n sampled completions (temperature 1, top-p 0.3, frequency penalty 1 — the paper's sampling settings).

    Without a KV cache every step re-runs the full sequence, so the hook adds coef·vec to the first positions each time,
    which is exactly ActAdd's "add to the first n_pos residual positions".
    """

    def hook(x, hook):
        x[:, : vec.shape[0]] += coef * vec
        return x

    hooks = [(f"blocks.{layer}.hook_resid_pre", hook)] if (vec is not None and coef != 0) else []
    outs = []
    for k in range(n):
        torch.manual_seed(seed + k)
        with model.hooks(fwd_hooks=hooks):
            o = model.generate(prompt, max_new_tokens=40, temperature=1.0, top_p=0.3, freq_penalty=1.0, verbose=False, use_past_kv_cache=False)
        outs.append(o[len(prompt) :])
    return outs


def compute(ctx) -> None:
    from transformers import pipeline

    t0 = time.time()
    model = hooked("gpt2-xl", device=ctx.device)
    sent = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=0 if ctx.device == "cuda" else -1)
    n = 3 if ctx.quick else 8
    results = {}
    g = torch.Generator().manual_seed(ctx.seed)
    for name, spec in VECTORS.items():
        L = spec["layer"]
        vec = steering_vector(model, spec["plus"], spec["minus"], L)
        rnd = torch.randn(vec.shape, generator=g).to(vec.device)
        rnd = rnd / rnd.norm() * vec.norm()
        rows = []
        for coef in COEFS if not ctx.quick else [0, 4]:
            for kind, v in (("steering vector", vec), ("random vector (control)", rnd)):
                if coef == 0 and kind != "steering vector":
                    continue
                comps = []
                for p in spec["prompts"]:
                    for c in generate(model, p, v, coef, L, n, ctx.seed):
                        comps.append({"prompt": p, "completion": c})
                        log.info("ActAdd %s c=%s %s | %s ||| %s", name, coef, kind, p, c.replace("\n", " "))
                texts = [x["completion"] for x in comps]
                wedding = float(np.mean([any(w in t.lower() for w in WEDDING_WORDS) for t in texts]))
                s = sent([x["prompt"] + x["completion"] for x in comps], truncation=True)
                pos = float(np.mean([r["label"] == "POSITIVE" for r in s]))
                rows.append({"coef": coef, "kind": kind, "wedding_rate": wedding, "positive_rate": pos, "examples": comps[: 2 * len(spec["prompts"])]})
                log.info("ActAdd %s c=%s %s: wedding %.2f positive %.2f", name, coef, kind, wedding, pos)
        results[name] = {"layer": L, "plus": spec["plus"], "minus": spec["minus"], "norm": float(vec.norm()), "rows": rows}
    w = results["weddings"]["rows"]
    lv = results["love vs hate"]["rows"]
    get = lambda rows, c, k, f: next((r[f] for r in rows if r["coef"] == c and r["kind"] == k), float("nan"))  # noqa: E731
    cmax = max(r["coef"] for r in w)
    summary = {
        "wedding_base": get(w, 0, "steering vector", "wedding_rate"),
        "wedding_c4": get(w, 4, "steering vector", "wedding_rate"),
        "wedding_rand_c4": get(w, 4, "random vector (control)", "wedding_rate"),
        "wedding_max": max(r["wedding_rate"] for r in w if r["kind"] == "steering vector"),
        "love_base": get(lv, 0, "steering vector", "positive_rate"),
        "love_c4": get(lv, 4, "steering vector", "positive_rate"),
        "love_rand_c4": get(lv, 4, "random vector (control)", "positive_rate"),
        "coef_max": cmax,
        "n_per_prompt": n,
    }
    repl = [
        verdict_row(
            "Adding a 'weddings' vector steers the topic",
            "> 90% of completions about weddings vs ~2% unsteered (Turner et al.)",
            f"wedding words in {summary['wedding_base'] * 100:.0f}% of completions unsteered → {summary['wedding_c4'] * 100:.0f}% at c = 4 (random vector {summary['wedding_rand_c4'] * 100:.0f}%); best {summary['wedding_max'] * 100:.0f}%",
            "c = 4 rate ≥ 50% and ≥ 30 points above both the unsteered and random-vector rates",
            summary["wedding_c4"] >= 0.5 and summary["wedding_c4"] - max(summary["wedding_base"], summary["wedding_rand_c4"]) >= 0.3,
            partial=summary["wedding_c4"] - max(summary["wedding_base"], summary["wedding_rand_c4"]) >= 0.15,
        ),
        verdict_row(
            "A 'Love − Hate' vector makes text more positive",
            "Love − Hate at layer 6 turns 'I hate you because' positive (Turner et al.)",
            f"positive sentiment {summary['love_base'] * 100:.0f}% → {summary['love_c4'] * 100:.0f}% at c = 4 (random vector {summary['love_rand_c4'] * 100:.0f}%)",
            "≥ 20 points above both baseline and random vector",
            summary["love_c4"] - max(summary["love_base"], summary["love_rand_c4"]) >= 0.2,
        ),
    ]
    write_entry(SLUG, {"vectors": results, "coefs": COEFS}, summary, repl, models=["gpt2-xl"], runtime_s=time.time() - t0, ndigits=3)
