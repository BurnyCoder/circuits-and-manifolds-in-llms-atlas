"""The Jacobian lens and the verbalizable "global workspace" (Gurnee et al. 2026), on Qwen3-1.7B.

Paper: Gurnee, Sofroniew, Pearce, Piotrowski, … Lindsey, "Verbalizable Representations Form a Global Workspace in
Language Models" (https://transformer-circuits.pub/2026/workspace/index.html; arXiv 2607.15495; blog
https://www.anthropic.com/research/global-workspace). The J-lens reads layer ℓ as softmax(W_U · norm(J_ℓ h_ℓ)) with
J_ℓ the Jacobian of a late-layer residual w.r.t. h_ℓ averaged over prompts and positions (logit lens = J_ℓ = I). In
Claude it surfaces unspoken intermediate steps (e.g. 21 → 42 → 49 in "(4+17)*2+7") in a middle "workspace" band, and
swapping J-lens coordinates (France ↔ China) changes downstream answers.
We use the pre-fitted Qwen3-1.7B lens from Neuronpedia (HF neuronpedia/jacobian-lens, fitted with Anthropic's
jacobian-lens library, Apache-2.0, on 466 WikiText prompts; target = final block), and report:
- slice grids (layer × position top tokens) for J-lens vs logit lens on multi-hop / arithmetic prompts;
- the best rank of the unspoken intermediate token (Texas, France) at the last position, J-lens vs logit lens (the
  arithmetic prompt is shown in the grids only: Qwen3 splits "21"/"42" into digit tokens);
- coordinate swaps at the country token in a middle layer: (a) pre-stated simple swap = exchange the projections on the
  two unit J-lens directions J_ℓᵀ W_U[t]; (b) paper-style swap = decompose the activation into a non-negative combination
  of the J-lens vectors of its top-25 J-lens tokens plus A and B (scipy NNLS) and exchange A's and B's coefficients;
  success = B's answer (capital/language) in the top 5; controls: a random direction / random vector of matched norm.
Note a published failed replication on GPT-2 small (J-lens worse than logit lens) and that selectivity did not
reproduce ≤ 9B (avalur/JLens).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from ..common.stats import rate_str

log = get_logger()
SLUG = "jlens"
MODEL = "Qwen/Qwen3-1.7B"
LENS = ("neuronpedia/jacobian-lens", "qwen3-1.7b/jlens/Salesforce-wikitext/Qwen3-1.7B_jacobian_lens.pt")
PROMPTS = {
    "dallas": ("Fact: The capital of the state containing Dallas is", [" Texas"], " Austin"),
    "paris": ("Fact: The language spoken in the country whose capital is Paris is", [" France"], " French"),
    "arith": ("Calculate: (4+17)*2+7 =", ["21", "42"], "49"),
    "eiffel": ("The Eiffel Tower is in the city of", [" France"], " Paris"),
}
# Qwen3 tokenizes numbers digit by digit ("21" -> "2", "1"; "42" and "49" both start with "4"), so multi-digit
# intermediates are not single tokens: the arithmetic prompt stays in the slice grids but not in the rank metric.
RANK_CASES = [k for k in PROMPTS if k != "arith"]
# Confirmatory test (methods review M11: the rank row above rests on n = 3 and is post-hoc). 24 fresh two-hop prompts,
# written before any lens readout on them; filtered only by tokenization (each intermediate is one Qwen3 token).
# Criterion pre-stated in reports/planned-runs.md ("J-lens confirmatory test on fresh prompts"), commit 8d1376e.
_STATE = "Fact: The capital of the state containing {} is"
_LANDMARK = "Fact: The capital of the country where the {} is located is"
_CAPITAL = "Fact: The language spoken in the country whose capital is {} is"
CONFIRM = (
    [(_STATE.format(c), " " + st) for c, st in [("Miami", "Florida"), ("Chicago", "Illinois"), ("Seattle", "Washington"), ("Detroit", "Michigan"),
                                               ("Las Vegas", "Nevada"), ("Philadelphia", "Pennsylvania"), ("Cleveland", "Ohio"), ("New Orleans", "Louisiana")]]
    + [(_LANDMARK.format(lm), " " + co) for lm, co in [("Colosseum", "Italy"), ("Kremlin", "Russia"), ("Acropolis", "Greece"), ("Taj Mahal", "India"),
                                                      ("Great Wall", "China"), ("Sagrada Familia", "Spain"), ("pyramids of Giza", "Egypt"), ("CN Tower", "Canada")]]
    + [(_CAPITAL.format(ca), " " + co) for ca, co in [("Berlin", "Germany"), ("Lisbon", "Portugal"), ("Tokyo", "Japan"), ("Moscow", "Russia"),
                                                     ("Stockholm", "Sweden"), ("Warsaw", "Poland"), ("Oslo", "Norway"), ("Ankara", "Turkey")]]
)
N_DICT = 25  # J-space: sparse non-negative combination of up to ~25 J-lens vectors (Gurnee et al. 2026)
SWAPS = [
    ("The capital of France is", " France", " China", " Paris", " Beijing"),
    ("The capital of Germany is", " Germany", " Japan", " Berlin", " Tokyo"),
    ("The capital of Italy is", " Italy", " Spain", " Rome", " Madrid"),
    ("People in France speak", " France", " Germany", " French", " German"),
    ("People in China speak", " China", " Japan", " Chinese", " Japanese"),
    ("The capital of Spain is", " Spain", " Italy", " Madrid", " Rome"),
    ("The capital of Japan is", " Japan", " Germany", " Tokyo", " Berlin"),
    ("People in Germany speak", " Germany", " France", " German", " French"),
]


def load_lens(device) -> dict[int, torch.Tensor]:
    """{layer: J_ℓ [d, d]} from the pre-fitted lens file (format probed at runtime; documented: J[ℓ] fp16 [d, d])."""
    from huggingface_hub import hf_hub_download

    obj = torch.load(hf_hub_download(*LENS), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "J" in obj:
        obj = obj["J"]
    if hasattr(obj, "state_dict") and not isinstance(obj, dict):
        obj = obj.state_dict()
    out = {}
    for k, v in obj.items():
        key = str(k).split(".")[-1]
        if torch.is_tensor(v) and v.ndim == 2 and key.lstrip("-").isdigit():
            out[int(key)] = v.to(device, torch.float32)
    assert out, f"unrecognised lens format: {type(obj)} {list(obj)[:5] if isinstance(obj, dict) else ''}"
    return out


def compute(ctx) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(ctx.device).eval()
    J = load_lens(ctx.device)
    nl = model.config.num_hidden_layers
    layers = sorted(J)
    log.info("J-lens loaded for layers %s…%s (%d maps)", layers[0], layers[-1], len(layers))
    norm, W_U = model.model.norm, model.lm_head.weight.float()

    @torch.no_grad()
    def readout(h: torch.Tensor, layer: int, lens: str) -> torch.Tensor:
        v = h.float() @ J[layer].T if lens == "jlens" else h.float()
        return (norm(v.to(torch.bfloat16)).float() @ W_U.T).log_softmax(-1)

    def tid(s):
        return tok(s, add_special_tokens=False).input_ids[0]

    grids, ranks = {}, {}
    for key, (text, inter, ans) in PROMPTS.items():
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        toks = [tok.decode([i]) for i in ids[0].tolist()]
        grid = {"jlens": [], "logit": []}
        rk = {"jlens": {}, "logit": {}}
        for lens in ("jlens", "logit"):
            for L in layers:
                lp = readout(hs[L][0], L, lens)  # [T, V]
                top = lp.topk(3, -1)
                grid[lens].append([[(tok.decode([int(i)]), float(p.exp())) for i, p in zip(top.indices[t], top.values[t])] for t in range(len(toks))])
                for w in (inter + [ans]) if key in RANK_CASES else []:
                    r = int((lp[-1] > lp[-1, tid(w)]).sum()) + 1
                    rk[lens].setdefault(w, []).append(r)
        grids[key] = {"text": text, "tokens": toks, "layers": layers, "grid": grid, "intermediates": inter, "answer": ans}
        ranks[key] = rk
        log.info("J-lens %s: best ranks %s | logit lens %s", key, {w: min(v) for w, v in rk["jlens"].items()}, {w: min(v) for w, v in rk["logit"].items()})

    # Confirmatory rank test: best rank over lens layers of the intermediate at the last position, per prompt.
    from scipy.stats import binomtest

    confirm = []
    for text, inter_tok in CONFIRM:
        assert len(tok(inter_tok, add_special_tokens=False).input_ids) == 1, inter_tok
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        best = {}
        for lens in ("jlens", "logit"):
            rs = []
            for L in layers:
                lp = readout(hs[L][0, -1:], L, lens)[0]
                rs.append(int((lp > lp[tid(inter_tok)]).sum()) + 1)
            best[lens] = min(rs)
        confirm.append({"prompt": text, "intermediate": inter_tok.strip(), "rank_j": best["jlens"], "rank_logit": best["logit"]})
        log.info("J-lens confirm | %s -> %s | best rank J %d vs logit %d", text, inter_tok.strip(), best["jlens"], best["logit"])
    wins_j = sum(c["rank_j"] < c["rank_logit"] for c in confirm)
    wins_l = sum(c["rank_logit"] < c["rank_j"] for c in confirm)
    sign_p = float(binomtest(wins_j, wins_j + wins_l, 0.5).pvalue) if wins_j + wins_l else 1.0  # two-sided sign test, ties dropped

    # Coordinate swaps at the country token in middle layers.
    def lens_dir(token: str, layer: int) -> torch.Tensor:
        """Residual direction at layer ℓ that raises token t in the J-lens readout: J_ℓᵀ W_U[t] (ignoring the norm's scale)."""
        d = J[layer].T @ W_U[tid(token)]
        return d / d.norm()

    from scipy.optimize import nnls

    def nnls_delta(h: torch.Tensor, layer: int, a: str, b: str, rand: torch.Tensor | None = None) -> torch.Tensor:
        """Paper-style swap: write h's J-space part as a non-negative combination of the J-lens vectors J_ℓᵀ W_U[t]
        of its top-25 J-lens tokens plus a and b (scipy NNLS), then exchange the coefficients of a and b.
        With ``rand``, b's vector is replaced by a random vector of the same norm (control)."""
        top = readout(h[None], layer, "jlens")[0].topk(N_DICT).indices.tolist()
        toks_ = list(dict.fromkeys(top + [tid(a), tid(b)]))
        V = torch.stack([J[layer].T @ W_U[t] for t in toks_], 1)  # [d, K]
        ia, ib = toks_.index(tid(a)), toks_.index(tid(b))
        if rand is not None:
            V[:, ib] = rand * V[:, ib].norm()
        beta, _ = nnls(V.double().cpu().numpy(), h.double().cpu().numpy())
        return float(beta[ib] - beta[ia]) * V[:, ia] + float(beta[ia] - beta[ib]) * V[:, ib]

    g = torch.Generator(device=ctx.device).manual_seed(ctx.seed)
    swap_layers = [L for L in layers if 0.3 * nl <= L <= 0.7 * nl][::2]
    swap = []
    for L in swap_layers:
        succ, succ_r, base_ok, succ_n, succ_nr = [], [], [], [], []
        for text, a, b, ans_a, ans_b in SWAPS:
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(ctx.device)
            pos = [i for i, t in enumerate(ids[0].tolist()) if t == tid(a)][-1]
            da, db = lens_dir(a, L), lens_dir(b, L)
            rnd = torch.randn(da.shape, generator=g, device=ctx.device)
            rnd = rnd / rnd.norm()

            def run(d1, d2, mode="coord"):
                def hook(m, i, o):
                    hs = o[0] if isinstance(o, tuple) else o
                    h = hs[0, pos].float()
                    if mode == "coord":
                        c1, c2 = h @ d1, h @ d2
                        hs[0, pos] = (h + (c2 - c1) * d1 + (c1 - c2) * d2).to(hs.dtype)  # exchange the two coordinates
                    else:  # mode == "nnls": d1 = (a, b), d2 = optional random vector
                        hs[0, pos] = (h + nnls_delta(h, L, d1[0], d1[1], d2)).to(hs.dtype)
                    return o

                hk = model.model.layers[L - 1].register_forward_hook(hook) if d1 is not None else None
                with torch.no_grad():
                    lg = model(ids).logits[0, -1]
                if hk:
                    hk.remove()
                return lg.float()

            base = run(None, None)
            base_ok.append(int(base.argmax()) == tid(ans_a))
            top5 = run(da, db).topk(5).indices.tolist()
            succ.append(tid(ans_b) in top5)
            succ_r.append(tid(ans_b) in run(da, rnd).topk(5).indices.tolist())
            succ_n.append(tid(ans_b) in run((a, b), None, "nnls").topk(5).indices.tolist())
            succ_nr.append(tid(ans_b) in run((a, b), rnd, "nnls").topk(5).indices.tolist())
        swap.append({"layer": L, "top5_success": float(np.mean(succ)), "random_top5": float(np.mean(succ_r)), "baseline_correct": float(np.mean(base_ok)), "n": len(SWAPS),
                     "nnls_top5": float(np.mean(succ_n)), "nnls_random_top5": float(np.mean(succ_nr))})
        log.info("J-lens NNLS swap at layer %d: target in top-5 %.2f (random vector %.2f)", L, swap[-1]["nnls_top5"], swap[-1]["nnls_random_top5"])
        log.info("J-lens swap at layer %d: target answer in top-5 %.2f (random direction %.2f; baseline correct %.2f)", L, swap[-1]["top5_success"], swap[-1]["random_top5"], swap[-1]["baseline_correct"])

    inter_j = [min(ranks[k]["jlens"][w]) for k in RANK_CASES for w in PROMPTS[k][1]]
    inter_l = [min(ranks[k]["logit"][w]) for k in RANK_CASES for w in PROMPTS[k][1]]
    best_swap = max(swap, key=lambda s: s["top5_success"])
    best_nnls = max(swap, key=lambda s: s["nnls_top5"])
    summary = {
        "n_layers_lens": len(layers),
        "inter_median_rank_j": float(np.median(inter_j)),
        "inter_median_rank_logit": float(np.median(inter_l)),
        "inter_top10_j": int(sum(r <= 10 for r in inter_j)),
        "inter_top10_logit": int(sum(r <= 10 for r in inter_l)),
        "n_inter": len(inter_j),
        "swap_layer": best_swap["layer"],
        "swap_top5": best_swap["top5_success"],
        "swap_random": best_swap["random_top5"],
        "swap_n": best_swap["n"],
        "nnls_layer": best_nnls["layer"],
        "nnls_top5": best_nnls["nnls_top5"],
        "nnls_random": best_nnls["nnls_random_top5"],
        "confirm_n": len(confirm),
        "confirm_wins_j": wins_j,
        "confirm_wins_logit": wins_l,
        "confirm_sign_p": sign_p,
        "confirm_median_rank_j": float(np.median([c["rank_j"] for c in confirm])),
        "confirm_median_rank_logit": float(np.median([c["rank_logit"] for c in confirm])),
        "confirm_top10_j": int(sum(c["rank_j"] <= 10 for c in confirm)),
        "confirm_top10_logit": int(sum(c["rank_logit"] <= 10 for c in confirm)),
    }
    wr = lambda rate, n: rate_str(round(rate * n), n)  # noqa: E731  rate with its Wilson 95% CI
    repl = [
        verdict_row(
            "The J-lens surfaces unspoken intermediate steps better than the logit lens",
            "J-lens beats logit / tuned lens; intermediates (21, 42) become readable in middle layers (Claude)",
            f"best rank of the intermediate token at the last position: median {summary['inter_median_rank_j']:.0f} (J-lens) vs {summary['inter_median_rank_logit']:.0f} (logit lens); "
            f"top-10 for {summary['inter_top10_j']}/{len(inter_j)} vs {summary['inter_top10_logit']}/{len(inter_l)}",
            "J-lens median rank lower than the logit lens's and ≥ half of intermediates reach the top 10",
            summary["inter_median_rank_j"] < summary["inter_median_rank_logit"] and summary["inter_top10_j"] >= len(inter_j) / 2,
            partial=summary["inter_median_rank_j"] < summary["inter_median_rank_logit"],
        ),
        verdict_row(
            "Swapping J-lens coordinates swaps the concept",
            "France → China swaps change capital/language/continent answers together (76/192 top-1 across functions, Claude)",
            f"layer {best_swap['layer']}: target answer in top-5 for {wr(best_swap['top5_success'], best_swap['n'])} of {best_swap['n']} prompts "
            f"(random direction {wr(best_swap['random_top5'], best_swap['n'])}; Wilson 95% CIs)",
            "≥ 50% and ≥ 30 points above the random-direction swap",
            best_swap["top5_success"] >= 0.5 and best_swap["top5_success"] - best_swap["random_top5"] >= 0.3,
            partial=best_swap["top5_success"] - best_swap["random_top5"] >= 0.2,
        ),
        verdict_row(
            "Swapping J-space coefficients (paper-style sparse NNLS decomposition) swaps the concept",
            "category (\"think of a {category}\") swap with the J-space component: 59% top-5 vs 5% for the non-J component (Claude)",
            f"layer {best_nnls['layer']}: target answer in top-5 for {wr(best_nnls['nnls_top5'], best_nnls['n'])} of {best_nnls['n']} prompts "
            f"(random vector {wr(best_nnls['nnls_random_top5'], best_nnls['n'])}; Wilson 95% CIs)",
            "≥ 50% and ≥ 30 points above the random-vector swap (amendment 2026-11-27, same thresholds)",
            best_nnls["nnls_top5"] >= 0.5 and best_nnls["nnls_top5"] - best_nnls["nnls_random_top5"] >= 0.3,
            partial=best_nnls["nnls_top5"] - best_nnls["nnls_random_top5"] >= 0.2,
        ),
        verdict_row(
            "Confirmatory: on 24 fresh two-hop prompts the J-lens ranks the unspoken intermediate better than the logit lens",
            "J-lens beats the logit lens at surfacing unspoken intermediates (Claude)",
            f"J-lens better on {wins_j}, logit lens better on {wins_l}, tied on {len(confirm) - wins_j - wins_l} of {len(confirm)} prompts "
            f"(sign test p = {sign_p:.3g}); median best rank {summary['confirm_median_rank_j']:.0f} vs {summary['confirm_median_rank_logit']:.0f}; "
            f"top-10 for {summary['confirm_top10_j']} vs {summary['confirm_top10_logit']}",
            "J-lens better on more prompts than the reverse and two-sided sign test p < 0.05 (pre-stated 2026-11-28 before running)",
            wins_j > wins_l and sign_p < 0.05,
            partial=wins_j > wins_l,
        ),
    ]
    write_entry(SLUG, {"grids": grids, "ranks": ranks, "swap": swap, "confirm": confirm, "n_layers": nl}, summary, repl, models=[MODEL, "neuronpedia/jacobian-lens"], runtime_s=time.time() - t0, ndigits=3)
    del model
    torch.cuda.empty_cache()
