"""Copy suppression: GPT-2 small head L10H7 suppresses tokens that earlier layers wanted to copy.

Recomputes the weight-based and behavioural evidence of McDougall, Conmy, Rushing, McGrath, Nanda 2023
(https://arxiv.org/abs/2310.04625): if earlier layers predict a token that already appears in the context, L10H7
attends back to that earlier occurrence and writes *negatively* onto the same token's logit.
- OV test: for each source token t, push its "effective embedding" (W_E + MLP0 output, the paper's MLP0(W_E)) through
  the head's OV circuit and the unembedding; rank the effect on logit(t) among all output tokens (most negative = 1).
  Paper: 84.70% of tokens have the diagonal entry among the 10 most negative (they rank within columns; we rank within
  rows — the "effect of attending to t on every logit" view — so numbers are comparable but not identical).
- QK test: with W_U[t] as query and the effective embedding of s as key, is the score highest for s = t?
  Paper: 95.72% diagonal-max.
- Behaviour: on prompts like "All's fair in love and", L10H7 attends to " love" and lowers its logit; zero-ablating
  L10H7 raises the probability of the copied token.
- Control: the same metrics for every other head in layers 10–11.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from ..common.models import hooked
from ..common.stats import rate_str

log = get_logger()
SLUG = "copy-suppression"
HEAD = (10, 7)

PROMPTS = [
    ("All's fair in love and", " love"),
    ("The cat chased the mouse, and then the dog chased the", " mouse"),
    ("She bought apples, oranges and more", " apples"),
    ("He said hello, and she said hello", " hello"),
    ("Paris is in France. Berlin is in Germany. Madrid is in", " France"),
    ("The password is secret. Never share the", " secret"),
    ("I like tea. My brother likes coffee. My sister likes", " tea"),
    ("Monday came before Tuesday, and Tuesday came before", " Tuesday"),
    # 16 more, written before running (reports/planned-runs.md, review follow-ups 2026-11-28).
    ("I went to the bank, and after the bank I went to the", " bank"),
    ("Alice gave Bob a book, and then Bob gave the book back to", " Bob"),
    ("The river flows into the sea, and the sea meets the", " sea"),
    ("We ate pizza on Friday and pizza again on", " pizza"),
    ("London is in England. Rome is in Italy. Dublin is in", " England"),
    ("My favourite colour is blue, and my sister's favourite colour is", " blue"),
    ("He opened the door, walked through the door, and closed the", " door"),
    ("Cats chase mice, dogs chase cats, and mice chase", " mice"),
    ("The teacher asked the student, and the student asked the", " student"),
    ("First comes winter, then spring, then summer, then", " spring"),
    ("She sang a song, then another song, and finally a", " song"),
    ("Coffee in the morning, coffee at noon, and tea in the", " morning"),
    ("The king and the queen met the", " king"),
    ("Water is wet and fire is", " wet"),
    ("Up is the opposite of down, and left is the opposite of", " down"),
    ("He likes dogs more than cats, but she likes cats more than", " cats"),
]


@torch.no_grad()
def effective_embedding(model, toks: torch.Tensor) -> torch.Tensor:
    """W_E[t] + MLP0(LN(W_E[t] + mean pos)) — attention in layer 0 approximated as attending to self (paper §3.2)."""
    x = model.W_E[toks] + model.W_pos.mean(0)
    # Approximate layer-0 attention by the token attending only to itself, as in the paper's effective embedding.
    attn = model.blocks[0].attn
    v = torch.einsum("td,hdk->thk", model.blocks[0].ln1(x[:, None])[:, 0], attn.W_V) + attn.b_V
    x = x + torch.einsum("thk,hkd->td", v, attn.W_O) + attn.b_O
    x = x + model.blocks[0].mlp(model.blocks[0].ln2(x[:, None]))[:, 0]
    return x


@torch.no_grad()
def ov_qk_stats(model, head, src: torch.Tensor, emb: torch.Tensor) -> dict:
    """OV diagonal ranks (row-wise) and QK diagonal-max fraction for one head over sampled tokens."""
    l, h = head
    blk = model.blocks[l]
    # OV: source effective embedding -> value -> output -> (final LN approx: divide by RMS norm) -> logits.
    xin = blk.ln1(emb[:, None])[:, 0]
    out = (xin @ blk.attn.W_V[h]) @ blk.attn.W_O[h]  # [n, d_model]
    out = out / out.norm(dim=-1, keepdim=True)  # final LayerNorm scale only rescales; rank within a row is unaffected
    logits = out @ model.W_U  # [n, vocab]
    diag = logits[torch.arange(len(src)), src]
    rank = (logits < diag[:, None]).sum(-1) + 1  # 1 = most negative
    vocab = logits.shape[1]
    # QK: query = unembedding of token t (what earlier layers predict), key = effective embedding of s (in context).
    q = model.W_U[:, src].T  # [n, d_model]
    q = q / q.norm(dim=-1, keepdim=True) * np.sqrt(model.cfg.d_model)  # treat as LN-normalised residual
    qh = q @ blk.attn.W_Q[h]
    kh = xin @ blk.attn.W_K[h]
    scores = qh @ kh.T  # [n_query, n_key]
    qk_diag_max = (scores.argmax(-1) == torch.arange(len(src), device=scores.device)).float().mean().item()
    return {
        "rank": rank.cpu().numpy(),
        "frac_top10": (rank <= 10).float().mean().item(),
        "frac_bottom5pct": (rank <= 0.05 * vocab).float().mean().item(),
        "qk_diag_max": qk_diag_max,
    }


@torch.no_grad()
def compute(ctx) -> None:
    t0 = time.time()
    model = hooked("gpt2", device=ctx.device)
    # Sample tokens: the 5,000 lowest ids skip-listed to word-like tokens (GPT-2 BPE ids roughly follow merge order).
    n = 1000 if ctx.quick else 5000
    src = torch.arange(300, 300 + n, device=ctx.device)
    emb = effective_embedding(model, src)
    main = ov_qk_stats(model, HEAD, src, emb)
    log.info("L10H7: OV top-10 %.3f, bottom-5%% %.3f, QK diag-max %.3f", main["frac_top10"], main["frac_bottom5pct"], main["qk_diag_max"])
    controls = {}
    for l in (10, 11):
        for h in range(model.cfg.n_heads):
            if (l, h) == HEAD:
                continue
            s = ov_qk_stats(model, (l, h), src, emb)
            controls[f"{l}.{h}"] = {k: s[k] for k in ("frac_top10", "frac_bottom5pct", "qk_diag_max")}
    ctrl_top10 = [c["frac_top10"] for c in controls.values()]

    # Behaviour on prompts: attention from the last token to the earlier copy, direct effect, ablation.
    examples = []

    def zero(z, hook):
        z[:, :, HEAD[1]] = 0.0
        return z

    for text, target in PROMPTS:
        toks = model.to_tokens(text)
        tid = model.to_single_token(target)
        logits, cache = model.run_with_cache(toks)
        p_clean = logits[0, -1].softmax(-1)[tid].item()
        p_abl = model.run_with_hooks(toks, fwd_hooks=[(f"blocks.{HEAD[0]}.attn.hook_z", zero)])[0, -1].softmax(-1)[tid].item()
        pat = cache["pattern", HEAD[0]][0, HEAD[1], -1].cpu().numpy()
        z = cache["z", HEAD[0]][0, -1, HEAD[1]]
        head_out = (z @ model.W_O[HEAD[0], HEAD[1]]) / cache["ln_final.hook_scale"][0, -1]
        direct = head_out @ model.W_U[:, tid]
        # Control token (review m5): the most probable next token that does not occur in the prompt.
        present = set(toks[0].tolist())
        ctrl = next(int(t) for t in logits[0, -1].argsort(descending=True).tolist() if int(t) not in present)
        direct_ctrl = (head_out @ model.W_U[:, ctrl]).item()
        str_toks = [model.tokenizer.decode([t]) for t in toks[0].tolist()]
        src_pos = [i for i, t in enumerate(toks[0].tolist()) if t == tid]
        examples.append(
            {
                "text": text,
                "target": target,
                "tokens": str_toks,
                "attn": pat,
                "copy_pos": src_pos,
                "attn_to_copy": float(pat[src_pos].sum()) if src_pos else 0.0,
                "direct_effect": direct.item(),
                "control_token": model.tokenizer.decode([ctrl]),
                "direct_control": direct_ctrl,
                "p_clean": p_clean,
                "p_ablated": p_abl,
            }
        )
        log.info("%-60s target %-10r attn %.2f direct %.2f p %.4f -> %.4f (ablated)", text, target, examples[-1]["attn_to_copy"], direct.item(), p_clean, p_abl)

    n_neg = sum(e["direct_effect"] < 0 for e in examples)
    n_below_ctrl = sum(e["direct_effect"] < e["direct_control"] for e in examples)
    n_up = sum(e["p_ablated"] > e["p_clean"] for e in examples)
    orig = examples[:8]  # the original pre-stated row keeps its original 8 prompts
    n_neg8, n_up8 = sum(e["direct_effect"] < 0 for e in orig), sum(e["p_ablated"] > e["p_clean"] for e in orig)
    hist, edges = np.histogram(np.log10(main["rank"]), bins=40, range=(0, np.log10(model.cfg.d_vocab)))
    summary = {
        "n_tokens": n,
        "ov_top10": main["frac_top10"],
        "ov_bottom5": main["frac_bottom5pct"],
        "qk_diag": main["qk_diag_max"],
        "ctrl_top10_median": float(np.median(ctrl_top10)),
        "ctrl_top10_max": float(np.max(ctrl_top10)),
        "ctrl_top10_argmax": max(controls, key=lambda k: controls[k]["frac_top10"]),
        "n_prompts": len(orig),
        "n_prompts_all": len(examples),
        "n_neg_direct": n_neg8,
        "n_ablation_up": n_up8,
        "n_neg_direct_all": n_neg,
        "n_ablation_up_all": n_up,
        "mean_attn_copy": float(np.mean([e["attn_to_copy"] for e in examples])),
        "n_below_control": n_below_ctrl,
    }
    repl = [
        verdict_row(
            "L10H7's OV circuit suppresses the token it attends to",
            "84.70% of tokens: diagonal among 10 most negative; 98.86% in bottom 5% (column ranking)",
            f"{main['frac_top10'] * 100:.1f}% in top-10 most negative; {main['frac_bottom5pct'] * 100:.1f}% in bottom 5% "
            f"(row ranking, {n} tokens); other L10–11 heads: median {np.median(ctrl_top10) * 100:.1f}%",
            "> 50% top-10 and > 5× the median other head in layers 10–11",
            main["frac_top10"] > 0.5 and main["frac_top10"] > 5 * max(np.median(ctrl_top10), 1e-3),
        ),
        verdict_row(
            "L10H7's QK circuit attends to tokens that are being predicted",
            "95.72% of rows have the diagonal as maximum",
            f"{main['qk_diag_max'] * 100:.1f}% diagonal-max over {n} tokens",
            "> 50% diagonal-max",
            main["qk_diag_max"] > 0.5,
            partial=main["qk_diag_max"] > 0.2,
        ),
        verdict_row(
            "On natural prompts L10H7 lowers the logit of a copied token",
            "worked examples, e.g. 'All's fair in love and' (Table 1)",
            f"negative direct effect on {n_neg8}/{len(orig)} prompts; ablation raises the copied token's probability on {n_up8}/{len(orig)}",
            "negative direct effect on ≥ 75% of prompts",
            n_neg8 >= 0.75 * len(orig),
        ),
        verdict_row(
            "Confirmatory: L10H7 suppresses the copied token specifically",
            "copy suppression lowers the logit of tokens that appear earlier in the context (McDougall et al.)",
            f"direct effect lower on the copied token than on a control token (most probable next token not in the prompt) on "
            f"{rate_str(n_below_ctrl, len(examples))} of {len(examples)} prompts; negative on {rate_str(n_neg, len(examples))} (Wilson 95% CIs)",
            "both ≥ 75% of the 24 prompts; partial if one holds (pre-stated 2026-11-28)",
            n_below_ctrl >= 0.75 * len(examples) and n_neg >= 0.75 * len(examples),
            partial=n_below_ctrl >= 0.75 * len(examples) or n_neg >= 0.75 * len(examples),
        ),
    ]
    payload = {"rank_hist": hist, "rank_edges": edges, "controls": controls, "examples": examples}
    write_entry(SLUG, payload, summary, repl, models=["gpt2"], runtime_s=time.time() - t0)
    log.info("copy suppression summary: %s", summary)
