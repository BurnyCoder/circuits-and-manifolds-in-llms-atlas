"""Induction heads: find them, test them causally, and watch them form during training.

What this recomputes (claims from Olsson et al. 2022, https://transformer-circuits.pub/2022/in-context-learning-and-
induction-heads/index.html and Elhage et al. 2021, https://transformer-circuits.pub/2021/framework/index.html):
1. A 2-layer attention-only transformer (TransformerLens ``attn-only-2l``) has a previous-token head in layer 0 and
   induction heads in layer 1; on repeated random tokens the second repeat becomes predictable (loss collapses).
2. Causal test: ablating those heads destroys the in-context copying (loss on the repeat goes back up), and a
   randomly initialised model with the same architecture has no such heads (control).
3. GPT-2 small has induction heads too (scores feed the GPT-2 Model Map; attention patterns for every head on two
   prompts are exported to ``gpt2-map.json``).
4. Phase change: across Pythia-160M training checkpoints (https://github.com/EleutherAI/pythia), the induction score
   and the in-context-learning (ICL) score jump together early in training (Olsson et al. Argument 1).
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from ..common.attention import head_scores, repeated_tokens
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger
from ..common.models import hooked
from ..common.stats import bootstrap_ci, ci_str

log = get_logger()
SLUG = "induction-heads"

# Pythia checkpoints (revision "step{n}"); each step = 1024 seqs x 2048 tokens = 2,097,152 tokens
# (Biderman et al. 2023, https://arxiv.org/abs/2304.01373, Table/§2).
PYTHIA_STEPS = [0, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 13000, 20000, 40000, 80000, 143000]
TOKENS_PER_STEP = 2_097_152


@torch.no_grad()
def _per_token_loss(model, tokens: torch.Tensor) -> torch.Tensor:
    """[batch, pos-1] next-token cross-entropy of a HookedTransformer."""
    logits = model(tokens)
    logp = logits[:, :-1].log_softmax(-1)
    return -logp.gather(-1, tokens[:, 1:, None])[..., 0]


@torch.no_grad()
def _scores_and_patterns(model, tokens: torch.Tensor, seq_len: int):
    """Per-layer head scores {prev, induction, duplicate} -> arrays [layers, heads]; plus cache of patterns."""
    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("hook_pattern"))
    out = {k: [] for k in ("prev", "induction", "duplicate")}
    for layer in range(model.cfg.n_layers):
        s = head_scores(cache["pattern", layer], seq_len)
        for k in out:
            out[k].append(s[k].cpu().numpy())
    return {k: np.stack(v) for k, v in out.items()}, cache


def _ablation_loss(model, tokens, heads: list[tuple[int, int]], seq_len: int) -> np.ndarray:
    """Per-sequence mean loss on the repeated half with the given heads' outputs zero-ablated (hook_z = 0)."""

    def zero_head(z, hook, head):
        z[:, :, head] = 0.0
        return z

    hooks = [(f"blocks.{l}.attn.hook_z", lambda z, hook, h=h: zero_head(z, hook, h)) for l, h in heads]
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    logp = logits[:, :-1].log_softmax(-1)
    loss = -logp.gather(-1, tokens[:, 1:, None])[..., 0]
    return loss[:, seq_len + 1 :].mean(-1).cpu().numpy()  # predictions of tokens 2..L of the repeat


def attn_only_2l(ctx) -> dict:
    """Head scores, example patterns, loss curves and ablations on attn-only-2l (+ random-init control)."""
    from transformer_lens import HookedTransformer

    model = hooked("attn-only-2l", device=ctx.device)
    L, B = 25, (16 if ctx.quick else 128)
    tok = repeated_tokens(B, L, 1000, 20000, model.tokenizer.bos_token_id, seed=ctx.seed).to(ctx.device)
    scores, cache = _scores_and_patterns(model, tok, L)
    loss = _per_token_loss(model, tok).cpu().numpy()  # [B, 2L]
    first, second = loss[:, :L].mean(-1), loss[:, L + 1 :].mean(-1)

    # Identify the strongest previous-token head (layer 0) and induction heads (layer 1, score > 0.3).
    prev_head = int(scores["prev"][0].argmax())
    ind_heads = [int(h) for h in np.where(scores["induction"][1] > 0.3)[0]]
    if not ind_heads:
        ind_heads = [int(scores["induction"][1].argmax())]
    log.info("attn-only-2l prev-token head 0.%d (%.2f); induction heads %s", prev_head, scores["prev"][0, prev_head], ind_heads)

    clean = second
    abl = {
        "clean": clean,
        "ablate induction heads": _ablation_loss(model, tok, [(1, h) for h in ind_heads], L),
        "ablate prev-token head": _ablation_loss(model, tok, [(0, prev_head)], L),
    }
    # Control: ablate the same number of random layer-1 heads that are NOT induction heads.
    rng = np.random.default_rng(ctx.seed)
    others = [h for h in range(model.cfg.n_heads) if h not in ind_heads]
    abl["ablate random L1 heads (control)"] = _ablation_loss(
        model, tok, [(1, int(h)) for h in rng.choice(others, size=min(len(ind_heads), len(others)), replace=False)], L
    )

    # Control: random-init model with identical architecture.
    # Pretrained TransformerLens configs carry init_weights=False; without re-enabling it the "random" model's attention
    # weights are all zero (found by the methods review). Copy the config and initialise properly.
    rand_cfg = copy.deepcopy(model.cfg)
    rand_cfg.init_weights = True
    rand = HookedTransformer(rand_cfg)
    rand.to(ctx.device).eval()
    rand_scores, _ = _scores_and_patterns(rand, tok, L)

    # Example sequence for the token-level viz: patterns of the prev-token head and the top induction head.
    ex = tok[0]
    toks = [model.tokenizer.decode([t]) for t in ex.tolist()]
    top_ind = int(scores["induction"][1].argmax())
    return {
        "scores": scores,
        "rand_scores": rand_scores,
        "first": first,
        "second": second,
        "loss_curve": loss.mean(0),
        "ablations": {k: bootstrap_ci(v) for k, v in abl.items()},
        "prev_head": prev_head,
        "ind_heads": ind_heads,
        "example": {
            "tokens": toks,
            "seq_len": L,
            "patterns": {
                f"0.{prev_head}": cache["pattern", 0][0, prev_head].cpu().numpy(),
                f"1.{top_ind}": cache["pattern", 1][0, top_ind].cpu().numpy(),
            },
        },
    }


def gpt2_small(ctx) -> dict:
    """GPT-2 small head scores on repeated tokens + attention patterns of all 144 heads on two prompts (Model Map)."""
    model = hooked("gpt2", device=ctx.device)
    L, B = 25, (16 if ctx.quick else 64)
    tok = repeated_tokens(B, L, 1000, 20000, model.tokenizer.bos_token_id, seed=ctx.seed).to(ctx.device)
    scores, _ = _scores_and_patterns(model, tok, L)
    prompts = {
        "ioi": "When Mary and John went to the store, John gave a drink to",
        "repeat": "Harry Potter waved his wand at the dragon. Harry Potter waved his wand at the",
    }
    patterns = {}
    for key, text in prompts.items():
        t = model.to_tokens(text, prepend_bos=True)
        _, cache = model.run_with_cache(t, names_filter=lambda n: n.endswith("hook_pattern"))
        patterns[key] = {
            "tokens": [model.tokenizer.decode([i]) for i in t[0].tolist()],
            "pattern": torch.stack([cache["pattern", l][0] for l in range(12)]).cpu().numpy(),  # [12,12,pos,pos]
        }
    return {"scores": scores, "patterns": patterns}


@torch.no_grad()
def pythia_phase_change(ctx) -> dict:
    """Max induction score and ICL score across Pythia-160M training checkpoints."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    steps = PYTHIA_STEPS if not ctx.quick else [0, 512, 2000, 8000, 143000]
    tokzr = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    # Natural text for the ICL score: WikiText-103 test split (https://huggingface.co/datasets/Salesforce/wikitext).
    text = "\n".join(load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"])
    ids = tokzr(text[:2_000_000], return_tensors="pt").input_ids[0]
    n_seq, ctx_len = (8 if ctx.quick else 128), 512
    nat = ids[: n_seq * ctx_len].view(n_seq, ctx_len).to(ctx.device)
    L = 25
    rep = repeated_tokens(16, L, 1000, 20000, tokzr.bos_token_id or 0, seed=ctx.seed).to(ctx.device)
    rows = []
    for step in steps:
        m = GPTNeoXForCausalLM.from_pretrained(
            "EleutherAI/pythia-160m", revision=f"step{step}", dtype=torch.float32, attn_implementation="eager"
        ).to(ctx.device)
        m.eval()
        out = m(rep, output_attentions=True)
        ind = np.stack([head_scores(a.float(), L)["induction"].cpu().numpy() for a in out.attentions])  # [layers, heads]
        lp = out.logits[:, :-1].float().log_softmax(-1)
        rep_loss = -lp.gather(-1, rep[:, 1:, None])[..., 0]
        # Chunk over sequences: full logits for 128 x 512 x 50k vocab would need ~13 GB.
        nll = torch.cat(
            [
                torch.nn.functional.cross_entropy(
                    m(c).logits[:, :-1].float().transpose(1, 2), c[:, 1:], reduction="none"
                )  # [chunk, ctx_len-1]; index t predicts token t+1
                for c in nat.split(8)
            ]
        )
        # Olsson et al.: ICL score = loss at the 500th token minus loss at the 50th token (averaged over contexts).
        # With far fewer contexts than the paper, single positions are noisy, so we average 11-token windows centred
        # on tokens 50 and 500 (indices 494..504 and 44..54 predict tokens 496..506 / 46..56) — a documented deviation.
        icl = (nll[:, 494:505].mean(-1) - nll[:, 44:55].mean(-1)).mean().item()
        rows.append(
            {
                "step": step,
                "tokens_B": step * TOKENS_PER_STEP / 1e9,
                "max_induction": float(ind.max()),
                "top_head": [int(x) for x in np.unravel_index(ind.argmax(), ind.shape)],
                "icl": icl,
                "rep_loss_first": rep_loss[:, :L].mean().item(),
                "rep_loss_second": rep_loss[:, L + 1 :].mean().item(),
            }
        )
        log.info("pythia-160m step %6d: max induction %.3f (L%d H%d), ICL %.3f", step, rows[-1]["max_induction"], *rows[-1]["top_head"], icl)
        del m
        torch.cuda.empty_cache()
    return {"rows": rows}


def compute(ctx) -> None:
    t0 = time.time()
    a2 = attn_only_2l(ctx)
    g2 = gpt2_small(ctx)
    py = pythia_phase_change(ctx)

    rows = py["rows"]
    jumps = [(rows[i + 1]["max_induction"] - rows[i]["max_induction"], i) for i in range(len(rows) - 1)]
    j, i = max(jumps)
    jump_lo, jump_hi = rows[i]["tokens_B"], rows[i + 1]["tokens_B"]
    ab = a2["ablations"]
    ind_max = float(a2["scores"]["induction"][1].max())
    rand_max = float(a2["rand_scores"]["induction"].max())
    summary = {
        "a2_prev_head": f"0.{a2['prev_head']}",
        "a2_prev_score": float(a2["scores"]["prev"][0, a2["prev_head"]]),
        "a2_ind_heads": ", ".join(f"1.{h}" for h in a2["ind_heads"]),
        "a2_ind_max": ind_max,
        "a2_rand_ind_max": rand_max,
        "a2_loss_first": float(a2["first"].mean()),
        "a2_loss_second": float(a2["second"].mean()),
        "a2_loss_abl_ind": ab["ablate induction heads"][0],
        "a2_loss_abl_prev": ab["ablate prev-token head"][0],
        "a2_loss_abl_rand": ab["ablate random L1 heads (control)"][0],
        "gpt2_n_ind_heads": int((g2["scores"]["induction"] > 0.3).sum()),
        "gpt2_top_ind": "{}.{}".format(*np.unravel_index(g2["scores"]["induction"].argmax(), (12, 12))),
        "gpt2_top_ind_score": float(g2["scores"]["induction"].max()),
        "pythia_jump_lo_B": jump_lo,
        "pythia_jump_hi_B": jump_hi,
        "pythia_final_ind": rows[-1]["max_induction"],
        "pythia_init_ind": rows[0]["max_induction"],
    }
    repl = [
        verdict_row(
            "A 2-layer attention-only model forms induction heads in layer 1",
            "Induction heads need ≥2 layers (Elhage et al. 2021); score defined by Olsson et al. 2022",
            f"max layer-1 induction score {ind_max:.2f} vs {rand_max:.3f} at random init",
            "trained max score > 0.3 and > 10× the random-init score",
            ind_max > 0.3 and ind_max > 10 * rand_max,
        ),
        verdict_row(
            "Induction heads causally implement in-context copying",
            "Ablating induction heads removes in-context learning in small attn-only models (Olsson et al., Argument 3)",
            f"repeat loss {ci_str(*ab['clean'])} → {ci_str(*ab['ablate induction heads'])} nats when ablated "
            f"(random-head control {ci_str(*ab['ablate random L1 heads (control)'])})",
            "ablation raises repeat loss by > 1 nat and > 3× the random-head control's increase",
            (ab["ablate induction heads"][0] - ab["clean"][0]) > 1
            and (ab["ablate induction heads"][0] - ab["clean"][0])
            > 3 * max(ab["ablate random L1 heads (control)"][0] - ab["clean"][0], 1e-3),
        ),
        verdict_row(
            "Induction requires a previous-token head (K-composition)",
            "Induction head keys read a previous-token head's output (Elhage et al. 2021)",
            f"ablating prev-token head 0.{a2['prev_head']} raises repeat loss to {ab['ablate prev-token head'][0]:.2f} nats",
            "ablation raises repeat loss by > 1 nat",
            (ab["ablate prev-token head"][0] - ab["clean"][0]) > 1,
        ),
        verdict_row(
            "Induction heads form abruptly early in training (phase change)",
            "ICL score jumps in a narrow window at ~2.5–5B tokens in Anthropic's models (Olsson et al., Argument 1)",
            f"Pythia-160M: largest jump in max induction score between {jump_lo:.2f}B and {jump_hi:.2f}B tokens "
            f"({rows[0]['max_induction']:.2f} at init → {rows[-1]['max_induction']:.2f} final)",
            "largest single-interval jump occurs before 20B tokens and accounts for > 50% of the total rise",
            jump_hi <= 20 and j > 0.5 * (max(r["max_induction"] for r in rows) - rows[0]["max_induction"]),
        ),
    ]
    payload = {
        "a2": {
            "scores": a2["scores"],
            "rand_scores": a2["rand_scores"],
            "loss_curve": a2["loss_curve"],
            "ablations": {k: list(v) for k, v in ab.items()},
            "example": a2["example"],
        },
        "gpt2": {"scores": g2["scores"]},
        "pythia": rows,
    }
    write_entry(SLUG, payload, summary, repl, models=["attn-only-2l", "gpt2", "EleutherAI/pythia-160m"], runtime_s=time.time() - t0)
    # GPT-2 attention patterns for the home-page Model Map live in their own file (shared by several entries).
    write_entry("gpt2-map", {"scores": g2["scores"], "patterns": g2["patterns"]}, {}, None, models=["gpt2"], ndigits=3)
    log.info("induction summary: %s", summary)
