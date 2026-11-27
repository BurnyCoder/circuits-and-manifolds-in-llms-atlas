"""The world inside a world model: a mountain-car video predictor whose frame embeddings trace a 1-D curve (trained here).

Goodfire, "The World Inside Neural Networks" (Geiger, Lubana, Fel, Merullo, Byun, Lewis, McGrath, 2026-05-07,
https://www.goodfire.com/research/the-world-inside-neural-networks); setup details in Wurgaft et al. 2026, App. B
(https://arxiv.org/abs/2605.05115): MountainCar-v0, 100 rollouts (≤ 200 steps) from a noisy-momentum policy (p = 0.7;
40% random actions) or a square-wave policy (p = 0.3; period 5–25); conv encoder → 64-d LayerNorm latent, 16-d action
embedding, GRU(64), residual conv decoder predicting the next frame. Frame embeddings form a smooth 1-D curve indexed by
the car's position; moving along a spline through position-binned means keeps decoded frames realistic, while straight
lines cut through "voids" (garbled frames) or teleport the car. Isometry over 50 anchors: arc-length r = 0.996 along
the manifold vs r = 0.06 for straight chords.
We train the same architecture at 64×64 resolution (plus an auxiliary decoder z → current frame so latents can be
visualized), probe position/velocity, fit a smoothing spline through 100 position-bin means, and compare manifold vs
chord paths (decoded frames, isometry). Control: the same analysis on the untrained encoder.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ..common.env import RESULTS, SITE_DATA, seed_everything
from ..common.geometry import pca, ridge_r2
from ..common.io import verdict_row, write_entry
from ..common.log import get_logger

log = get_logger()
SLUG = "world-model"
S = 64


def rollouts(n: int, rng) -> list[dict]:
    import gymnasium as gym

    env = gym.make("MountainCar-v0", render_mode="rgb_array")
    out = []
    for ep in range(n):
        obs, _ = env.reset(seed=int(rng.integers(1e9)))
        square = rng.random() < 0.3
        period = int(rng.integers(5, 26))
        frames, acts, pos, vel = [], [], [], []
        for t in range(200):
            if square:
                a = 2 if (t // period) % 2 == 0 else 0
            else:
                a = 2 if obs[1] > 0 else 0
                if rng.random() < 0.4:
                    a = int(rng.integers(0, 3))
            img = Image.fromarray(env.render()).resize((S, S), Image.BILINEAR)
            frames.append(np.asarray(img, dtype=np.uint8))
            acts.append(a)
            pos.append(obs[0])
            vel.append(obs[1])
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        out.append({"frames": np.stack(frames), "acts": np.array(acts), "pos": np.array(pos), "vel": np.array(vel)})
    env.close()
    return out


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(), nn.Linear(64 * 8 * 8, 64), nn.LayerNorm(64),
        )
        self.act = nn.Embedding(3, 16)
        self.gru = nn.GRU(80, 64, batch_first=True)

        def decoder():
            return nn.Sequential(
                nn.Linear(64, 64 * 8 * 8), nn.Unflatten(1, (64, 8, 8)), nn.ReLU(),
                nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(32, 3, 4, 2, 1),
            )

        self.dec_next = decoder()  # residual next-frame prediction from the GRU state
        self.dec_now = decoder()  # auxiliary: reconstruct the current frame from its latent (for visualization)

    def encode(self, x):
        return self.enc(x)

    def forward(self, frames, acts):
        B, T = frames.shape[:2]
        z = self.encode(frames.flatten(0, 1)).view(B, T, 64)
        h, _ = self.gru(torch.cat([z, self.act(acts)], -1))
        nxt = frames[:, :-1].flatten(0, 1) + self.dec_next(h[:, :-1].flatten(0, 1))
        now = self.dec_now(z.flatten(0, 1))
        return z, nxt.view(B, T - 1, 3, S, S), now.view(B, T, 3, S, S)


def save_strip(frames: np.ndarray, path) -> None:
    """Horizontal strip of decoded frames ([K, 3, S, S] in [0, 1]) as one PNG."""
    imgs = (np.clip(frames, 0, 1) * 255).astype(np.uint8).transpose(0, 2, 3, 1)
    Image.fromarray(np.concatenate(list(imgs), 1)).save(path)


def compute(ctx) -> None:
    from scipy.interpolate import make_smoothing_spline

    t0 = time.time()
    seed_everything(ctx.seed)
    rng = np.random.default_rng(ctx.seed)
    data = rollouts(20 if ctx.quick else 100, rng)
    log.info("world model: %d rollouts, %d frames rendered in %.0fs", len(data), sum(len(d["acts"]) for d in data), time.time() - t0)
    model = WorldModel().to(ctx.device)
    untrained = WorldModel().to(ctx.device)
    untrained.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    T = 16
    steps = 300 if ctx.quick else 4000
    for step in range(steps):
        seqs = [d for d in data if len(d["acts"]) > T]
        batch = [seqs[i] for i in rng.integers(0, len(seqs), 32)]
        st = [int(rng.integers(0, len(d["acts"]) - T)) for d in batch]
        fr = torch.tensor(np.stack([d["frames"][s : s + T] for d, s in zip(batch, st)]), device=ctx.device).permute(0, 1, 4, 2, 3).float() / 255
        ac = torch.tensor(np.stack([d["acts"][s : s + T] for d, s in zip(batch, st)]), device=ctx.device)
        _, nxt, now = model(fr, ac)
        loss = ((nxt - fr[:, 1:]) ** 2).mean() + ((now - fr) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 500 == 0:
            log.info("world model step %d loss %.5f", step, loss.item())
    (RESULTS / SLUG).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), RESULTS / SLUG / "mountain_car_world_model.pt")
    log.info("world model: trained %d steps (%.0fs elapsed)", steps, time.time() - t0)

    allf = np.concatenate([d["frames"] for d in data])
    rid = np.concatenate([np.full(len(d["acts"]), i) for i, d in enumerate(data)])  # rollout id of every frame
    pos = np.concatenate([d["pos"] for d in data])
    vel = np.concatenate([d["vel"] for d in data])

    @torch.no_grad()
    def latents(m):
        out = []
        for i in range(0, len(allf), 512):
            x = torch.tensor(allf[i : i + 512], device=ctx.device).permute(0, 3, 1, 2).float() / 255
            out.append(m.encode(x).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def gru_states(m):
        """GRU state after each frame of each rollout (the model's memory, which can carry velocity)."""
        out = []
        for d in data:
            fr = torch.tensor(d["frames"], device=ctx.device).permute(0, 3, 1, 2).float() / 255
            z = m.encode(fr)[None]
            h, _ = m.gru(torch.cat([z, m.act(torch.tensor(d["acts"], device=ctx.device))[None]], -1))
            out.append(h[0].cpu().numpy())
        return np.concatenate(out)

    Z, Zu = latents(model), latents(untrained)
    Hs, Hu = gru_states(model), gru_states(untrained)
    # Split by rollout: neighbouring frames of one rollout are near-duplicates, so a per-frame split would leak.
    tr = rid % 5 != 0
    probes = {
        "position (trained)": ridge_r2(Z[tr], pos[tr, None], Z[~tr], pos[~tr, None]),
        "velocity (trained)": ridge_r2(Z[tr], vel[tr, None], Z[~tr], vel[~tr, None]),
        "position (untrained encoder)": ridge_r2(Zu[tr], pos[tr, None], Zu[~tr], pos[~tr, None]),
        # Extra (not a pre-stated verdict): one frame cannot show velocity; the recurrent state can.
        "velocity from GRU state (trained)": ridge_r2(Hs[tr], vel[tr, None], Hs[~tr], vel[~tr, None]),
        "velocity from GRU state (untrained)": ridge_r2(Hu[tr], vel[tr, None], Hu[~tr], vel[~tr, None]),
    }
    log.info("world model probes: %s", probes)
    # Manifold: smoothing spline (per latent coordinate) through 100 position-bin means.
    edges = np.quantile(pos, np.linspace(0, 1, 101))
    centers, means = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (pos >= a) & (pos <= b)
        if m.sum() >= 3:
            centers.append(pos[m].mean())
            means.append(Z[m].mean(0))
    centers, means = np.array(centers), np.stack(means)
    order = np.argsort(centers)
    centers, means = centers[order], means[order]
    splines = [make_smoothing_spline(centers, means[:, k], lam=None) for k in range(64)]
    curve = lambda p: np.stack([s(p) for s in splines], -1)  # noqa: E731
    # Isometry over 50 anchors: arc length along the spline / chord length vs true position difference.
    anchors = np.linspace(-0.9, 0.4, 50)
    fine = np.linspace(-1.1, 0.5, 2000)
    cf = curve(fine)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cf, axis=0), axis=1))]
    arc_at = np.interp(anchors, fine, arc)
    pa = curve(anchors)
    i, j = np.triu_indices(50, 1)
    dpos = np.abs(anchors[i] - anchors[j])
    r_arc = float(np.corrcoef(np.abs(arc_at[i] - arc_at[j]), dpos)[0, 1])
    r_chord = float(np.corrcoef(np.linalg.norm(pa[i] - pa[j], axis=1), dpos)[0, 1])
    # Decode frames along the manifold vs the straight chord between two positions.
    p0, p1 = -0.9, 0.3
    K = 10
    ts = np.linspace(0, 1, K)
    man = curve(p0 + (p1 - p0) * ts)
    lin = (1 - ts)[:, None] * curve(np.array([p0]))[0] + ts[:, None] * curve(np.array([p1]))[0]
    with torch.no_grad():
        dec = lambda z: model.dec_now(torch.tensor(z, dtype=torch.float32, device=ctx.device)).cpu().numpy()  # noqa: E731
        fm, fl = dec(man), dec(lin)
    img_dir = SITE_DATA.parent / "img" / "world-model"
    img_dir.mkdir(parents=True, exist_ok=True)
    save_strip(fm, img_dir / "manifold_path.png")
    save_strip(fl, img_dir / "linear_path.png")
    # Distance of each path point to the nearest real latent (off-manifold "voids").
    def void(zs):
        return float(np.mean([np.min(np.linalg.norm(Z[::5] - z, axis=1)) for z in zs]))

    sc, _, var, _ = pca(Z[::3], 3)
    summary = {
        "r2_position": probes["position (trained)"],
        "r2_velocity": probes["velocity (trained)"],
        "r2_position_untrained": probes["position (untrained encoder)"],
        "r2_velocity_gru": probes["velocity from GRU state (trained)"],
        "r2_velocity_gru_untrained": probes["velocity from GRU state (untrained)"],
        "r_arc": r_arc,
        "r_chord": r_chord,
        "void_manifold": void(man),
        "void_linear": void(lin),
        "n_frames": int(len(Z)),
        "var3": float(var.sum()),
    }
    repl = [
        verdict_row(
            "The world model's latents encode the car's state",
            "linear probes: position R² ≈ 0.95, velocity ≈ 0.90 (Wurgaft et al. App. B)",
            f"position R² {probes['position (trained)']:.2f}, velocity {probes['velocity (trained)']:.2f} (untrained encoder: position {probes['position (untrained encoder)']:.2f})",
            "position R² ≥ 0.8 and ≥ 0.2 above the untrained encoder",
            probes["position (trained)"] >= 0.8 and probes["position (trained)"] - probes["position (untrained encoder)"] >= 0.2,
        ),
        verdict_row(
            "Distances along the manifold match distances in the world; straight lines don't",
            "arc-length r = 0.996 (manifold) vs 0.06 (chord)",
            f"r = {r_arc:.3f} along the spline vs {r_chord:.3f} for straight chords (50 anchors, 1,225 pairs)",
            "manifold r ≥ 0.9 and > chord r",
            r_arc >= 0.9 and r_arc > r_chord,
            partial=r_arc > r_chord,
        ),
    ]
    payload = {
        "pcs": sc,
        "pos": pos[::3],
        "manifold_img": "img/world-model/manifold_path.png",
        "linear_img": "img/world-model/linear_path.png",
        "path_positions": (p0 + (p1 - p0) * ts),
        "probes": probes,
    }
    write_entry(SLUG, payload, summary, repl, models=["mountain-car world model (trained here)"], runtime_s=time.time() - t0, ndigits=3)
