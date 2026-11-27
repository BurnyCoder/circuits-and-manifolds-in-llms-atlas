"""Neurons as an interpretable basis: how sparse are SwiGLU neurons per token? (Goodfire 2026, small-scale replication)

Goodfire ("Neurons as an Interpretable Basis", Tigges, Kowal, McGrath, Lewis, Cammarata, 2026-11-10,
https://www.goodfire.com/research/neurons-as-an-interpretable-basis) zero all but the top 320 of 4,096 routed neuron
activations per token in every layer of Qwen3.6-35B-A3B (92.2% ablated): web-text loss 1.886 → 1.940 (KL 0.140), while
keeping a random 64 of 512 per expert gives loss 4.214. They also split SwiGLU activations a = SiLU(g)·u into sign
quadrants (Q1 g>0,u>0; Q2 g>0,u<0; Q3 g<0,u<0; Q4 g<0,u>0) and find the most causally important cells are gate-open
(Q1/Q2), although 71–72% of all activations are gate-closed (Q3/Q4).
(Their layer top-320 keeps neurons "by absolute activation strength" and zeroes the rest before the down projection;
their percentile-band experiments mean-ablate.)
Here (dense Qwen3-1.7B, 28 layers × 6,144 neurons): per-token top-k over |a| before the down projection in every layer,
for k from 6,144 down to 64 (k = 480 ≙ 92.2% ablated), loss/KL on held-out WikiText-103; controls: random-k and
"drop only the top-480"; P25–P75 band mean-ablation; quadrant census and gradient×activation attribution by quadrant.
Comparable open replication: ericzhou12/neuron-quadrant-interventions (Qwen3-1.7B, KL ≈ 0.80 at 92.2%, with a different
rule: keep the top-k by distance from each neuron's mean and mean-fill the rest). We also evaluate that rule at k = 480
(post-hoc comparison, reported separately from the pre-stated verdicts).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "neuron-basis"
MODEL = "Qwen/Qwen3-1.7B"
KS = [6144, 3072, 1536, 768, 480, 256, 128, 64]


class Sparsifier:
    """Forward pre-hooks on every MLP's down_proj: transform its input a = SiLU(gate(x)) * up(x) per token."""

    def __init__(self, model):
        self.model, self.mode, self.k, self.band, self.mu, self.handles = model, None, None, None, None, []
        for li, layer in enumerate(model.model.layers):
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self._hook(li)))

    def _hook(self, li):
        def fn(mod, args):
            if self.mode is None:
                return None
            a = args[0]
            if self.mode == "topk":
                idx = a.abs().topk(self.k, dim=-1).indices
                return (torch.zeros_like(a).scatter(-1, idx, a.gather(-1, idx)),)
            if self.mode == "drop_topk":
                idx = a.abs().topk(self.k, dim=-1).indices
                return (a.scatter(-1, idx, 0.0),)
            if self.mode == "random":
                r = torch.rand_like(a, dtype=torch.float32)
                idx = r.topk(self.k, dim=-1).indices
                return (torch.zeros_like(a).scatter(-1, idx, a.gather(-1, idx)),)
            if self.mode == "topk_mean":
                # ericzhou12's rule (post-hoc comparison): keep the k neurons farthest from their mean, mean-fill the rest.
                mu = self.mu[li].to(a.dtype)
                idx = (a - mu).abs().topk(self.k, dim=-1).indices
                return (mu.expand_as(a).clone().scatter(-1, idx, a.gather(-1, idx)),)
            if self.mode == "band":
                lo, hi, mean = self.band[li]
                inside = (a >= lo) & (a <= hi)
                return (torch.where(inside, mean.to(a.dtype).expand_as(a), a),)
            return None

        return fn


@torch.no_grad()
def eval_loss(model, seqs, ref_logp=None):
    """Mean next-token loss and (optionally) mean KL(ref || model) over sequences."""
    losses, kls, logps = [], [], []
    for s in seqs:
        logits = model(s[None]).logits[0, :-1].float()
        lp = logits.log_softmax(-1)
        losses.append(-lp.gather(-1, s[1:, None])[:, 0].mean().item())
        if ref_logp is not None:
            ref = ref_logp[len(logps)]
            kls.append((ref.exp() * (ref - lp)).sum(-1).mean().item())
        logps.append(lp.half() if ref_logp is None else None)
    return float(np.mean(losses)), (float(np.mean(kls)) if kls else 0.0), logps


def compute(ctx) -> None:
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(ctx.device).eval()
    text = "\n\n".join(t for t in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")["text"] if t.strip())
    ids = tok(text[:600_000], return_tensors="pt").input_ids[0]
    n_seq, L = (6, 256) if ctx.quick else (48, 512)
    seqs = [ids[i * L : (i + 1) * L].to(ctx.device) for i in range(n_seq)]
    sp = Sparsifier(model)

    base_loss, _, ref = eval_loss(model, seqs)
    ref = [r.float() for r in ref]
    curve = []
    for k in KS[1:] if not ctx.quick else [1536, 480, 128]:
        sp.mode, sp.k = "topk", k
        loss, kl, _ = eval_loss(model, seqs, ref)
        curve.append({"k": k, "frac_ablated": 1 - k / 6144, "loss": loss, "kl": kl})
        log.info("top-%d per token per layer (%.1f%% ablated): loss %.4f (base %.4f), KL %.4f", k, 100 * (1 - k / 6144), loss, base_loss, kl)
    controls = {}
    for mode in ("random", "drop_topk"):
        sp.mode, sp.k = mode, 480
        controls[mode] = dict(zip(("loss", "kl"), eval_loss(model, seqs, ref)[:2]))
        log.info("control %s-480: %s", mode, controls[mode])

    # Band ablation (P25–P75 mean-ablated) with per-neuron statistics from separate calibration text.
    sp.mode = None
    calib = ids[n_seq * L : n_seq * L + 8 * L].to(ctx.device).view(8, L)
    store = {}
    hooks = [layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, li=li: store.setdefault(li, []).append(a[0].float().reshape(-1, a[0].shape[-1])[::4].cpu())) for li, layer in enumerate(model.model.layers)]
    with torch.no_grad():
        for c in calib:
            model(c[None])
    for h in hooks:
        h.remove()
    band = {}
    for li, chunks in store.items():
        A = torch.cat(chunks)
        q = torch.quantile(A[: min(len(A), 4000)], torch.tensor([0.25, 0.75]), dim=0)
        inside = (A >= q[0]) & (A <= q[1])
        mean = (A * inside).sum(0) / inside.sum(0).clamp_min(1)
        band[li] = (q[0].to(ctx.device, torch.bfloat16), q[1].to(ctx.device, torch.bfloat16), mean.to(ctx.device))
    sp.mu = {li: torch.cat(chunks).mean(0).to(ctx.device) for li, chunks in store.items()}
    sp.mode, sp.k = "topk_mean", 480
    controls["topk_mean_480"] = dict(zip(("loss", "kl"), eval_loss(model, seqs, ref)[:2]))
    log.info("post-hoc: top-480 by distance from mean, mean-filled (ericzhou12 rule): %s", controls["topk_mean_480"])
    sp.mode, sp.band = "band", band
    controls["band_p25_p75"] = dict(zip(("loss", "kl"), eval_loss(model, seqs, ref)[:2]))
    sp.mode = None
    log.info("band P25–P75 mean-ablation: %s", controls["band_p25_p75"])

    # Quadrant census + gradient×activation attribution by quadrant (top 1% cells per sequence).
    acts = {}

    def cap(li):
        def fn(mod, inp, out):
            x = inp[0]
            g = mod.gate_proj(x)
            u = mod.up_proj(x)
            acts[li] = (g.detach(), u.detach())
        return fn

    hh = [layer.mlp.register_forward_hook(cap(li)) for li, layer in enumerate(model.model.layers)]
    a_store = {}
    hd = [layer.mlp.down_proj.register_forward_pre_hook(lambda m, a, li=li: (a[0].retain_grad(), a_store.__setitem__(li, a[0]))[1]) for li, layer in enumerate(model.model.layers)]
    census = np.zeros((len(model.model.layers), 4))
    attr_q = np.zeros(4)
    mag_q = np.zeros(4)
    for s in seqs[: (2 if ctx.quick else 8)]:
        model.zero_grad()
        with torch.enable_grad():
            out = model(s[None])
            loss = torch.nn.functional.cross_entropy(out.logits[0, :-1].float(), s[1:])
            loss.backward()
        cells = []
        for li in range(len(model.model.layers)):
            g, u = acts[li]
            q = torch.where(g > 0, torch.where(u > 0, 0, 1), torch.where(u < 0, 2, 3))[0]  # Q1..Q4 -> 0..3
            census[li] += np.bincount(q.flatten().cpu().numpy(), minlength=4)
            a = a_store[li]
            at = (a * a.grad)[0].abs().float()
            mag_q += np.array([a[0].abs().float()[q == i].sum().item() for i in range(4)])
            cells.append((at.flatten(), q.flatten()))
        allat = torch.cat([c[0] for c in cells])
        allq = torch.cat([c[1] for c in cells])
        top = allat.topk(max(1, int(0.01 * len(allat)))).indices
        attr_q += np.bincount(allq[top].cpu().numpy(), minlength=4)
    for h in hh + hd:
        h.remove()
    census_frac = census / census.sum(1, keepdims=True)
    attr_frac = attr_q / attr_q.sum()
    all_frac = census.sum(0) / census.sum()
    k480 = next(c for c in curve if c["k"] == 480)
    summary = {
        "base_loss": base_loss,
        "k480_loss": k480["loss"],
        "k480_rel": k480["loss"] / base_loss - 1,
        "k480_kl": k480["kl"],
        "random480_loss": controls["random"]["loss"],
        "drop480_loss": controls["drop_topk"]["loss"],
        "band_loss": controls["band_p25_p75"]["loss"],
        "band_kl": controls["band_p25_p75"]["kl"],
        "k480_meanrule_kl": controls["topk_mean_480"]["kl"],  # post-hoc: ericzhou12's selection rule
        "k480_meanrule_loss": controls["topk_mean_480"]["loss"],
        "gate_closed_frac": float(all_frac[2] + all_frac[3]),
        "top_attr_gate_open": float(attr_frac[0] + attr_frac[1]),
        "n_tokens": n_seq * (L - 1),
    }
    repl = [
        verdict_row(
            "Most neuron activations can be zeroed per token with little damage",
            "keep 320/4,096 (92.2% ablated): loss 1.886 → 1.940 (+2.9%), KL 0.140 (Qwen3.6-35B-A3B MoE); open Qwen3-1.7B replication: KL ≈ 0.80",
            f"keep 480/6,144 (92.2% ablated): loss {base_loss:.3f} → {k480['loss']:.3f} ({summary['k480_rel'] * 100:+.1f}%), KL {k480['kl']:.3f}; random-480 → {controls['random']['loss']:.3f}",
            "relative loss increase < 25% and KL < 1.0, and the random-480 control at least 3× worse in loss increase",
            summary["k480_rel"] < 0.25 and k480["kl"] < 1.0 and (controls["random"]["loss"] - base_loss) > 3 * (k480["loss"] - base_loss),
            partial=(controls["random"]["loss"] - base_loss) > 3 * (k480["loss"] - base_loss),
        ),
        verdict_row(
            "The few large activations carry the computation",
            "ablating only the top 12.5% per expert → loss 4.587 (bottom-448 control)",
            f"zeroing only the top 480 per token → loss {controls['drop_topk']['loss']:.3f} (base {base_loss:.3f})",
            "loss increase ≥ 5× that of keeping only the top 480",
            (controls["drop_topk"]["loss"] - base_loss) >= 5 * max(k480["loss"] - base_loss, 1e-3),
        ),
        verdict_row(
            "Gate-open quadrants carry causal weight although most activations are gate-closed",
            "top attributed cells: Q1 39–50% + Q2 43–57%; 71–72% of all activations in Q3/Q4",
            f"top-1% gradient×activation cells: {summary['top_attr_gate_open'] * 100:.0f}% gate-open (Q1+Q2); all activations: {summary['gate_closed_frac'] * 100:.0f}% gate-closed",
            "≥ 80% of top cells gate-open and ≥ 50% of all activations gate-closed",
            summary["top_attr_gate_open"] >= 0.8 and summary["gate_closed_frac"] >= 0.5,
            partial=summary["top_attr_gate_open"] >= 0.6,
        ),
    ]
    payload = {"curve": [{"k": 6144, "frac_ablated": 0.0, "loss": base_loss, "kl": 0.0}] + curve, "controls": controls, "census": census_frac, "attr_frac": attr_frac, "mag_frac": mag_q / mag_q.sum(), "all_frac": all_frac}
    write_entry(SLUG, payload, summary, repl, models=[MODEL], runtime_s=time.time() - t0, ndigits=4)
    del model
    torch.cuda.empty_cache()
