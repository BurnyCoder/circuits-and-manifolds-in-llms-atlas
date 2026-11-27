// Toy Models of Superposition, trained live in your browser (Elhage et al. 2022,
// https://transformer-circuits.pub/2022/toy_model/index.html): n = 5 sparse features squeezed into m = 2 dimensions
// by x' = ReLU(WᵀW x + b), with importance-weighted MSE loss Σᵢ Iᵢ (xᵢ − x'ᵢ)². Each feature is present with probability
// 1 − S (value ~ U[0, 1]); importance Iᵢ = decayⁱ. Low sparsity → only the 2 most important features get dimensions;
// high sparsity → all 5 share the plane as a pentagon (superposition). Gradients are hand-derived; Adam optimizer.
import { useEffect, useRef, useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import { HowTo, Kpis } from '../primitives/common'

const N = 5
const M = 2
const STEPS = 4000
const BATCH = 256

function mulberry32(seed: number) {
  return () => {
    seed |= 0
    seed = (seed + 0x6d2b79f5) | 0
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface State {
  W: number[][] // [M][N]
  b: number[]
  loss: number
  step: number
}

function useToyModel(density: number, decay: number, seed: number) {
  const [state, setState] = useState<State | null>(null)
  const raf = useRef(0)
  useEffect(() => {
    const rnd = mulberry32(seed)
    const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd())
    const W = [...Array(M)].map(() => [...Array(N)].map(() => gauss() * 0.5))
    const b = Array(N).fill(0)
    const I = [...Array(N)].map((_, i) => decay ** i)
    // Adam state
    const mW = W.map((r) => r.map(() => 0))
    const vW = W.map((r) => r.map(() => 0))
    const mb = b.map(() => 0)
    const vb = b.map(() => 0)
    const lr = 0.01
    const [b1, b2, eps] = [0.9, 0.999, 1e-8]
    let step = 0
    const tick = () => {
      let loss = 0
      for (let it = 0; it < 40 && step < STEPS; it++, step++) {
        const gW = W.map((r) => r.map(() => 0))
        const gb = Array(N).fill(0)
        loss = 0
        for (let s = 0; s < BATCH; s++) {
          const x = [...Array(N)].map(() => (rnd() < density ? rnd() : 0))
          const h = [...Array(M)].map((_, k) => W[k].reduce((acc, w, j) => acc + w * x[j], 0))
          const y = [...Array(N)].map((_, i) => b[i] + h.reduce((acc, hk, k) => acc + W[k][i] * hk, 0))
          const out = y.map((v) => Math.max(0, v))
          const gy = out.map((o, i) => (y[i] > 0 ? (2 * I[i] * (o - x[i])) / BATCH : 0))
          out.forEach((o, i) => (loss += (I[i] * (o - x[i]) ** 2) / BATCH))
          for (let i = 0; i < N; i++) gb[i] += gy[i]
          for (let k = 0; k < M; k++) {
            let gh = 0
            for (let i = 0; i < N; i++) {
              gW[k][i] += gy[i] * h[k] // through the readout Wᵀ
              gh += gy[i] * W[k][i]
            }
            for (let j = 0; j < N; j++) gW[k][j] += gh * x[j] // through the encoder W
          }
        }
        const t = step + 1
        for (let k = 0; k < M; k++)
          for (let i = 0; i < N; i++) {
            mW[k][i] = b1 * mW[k][i] + (1 - b1) * gW[k][i]
            vW[k][i] = b2 * vW[k][i] + (1 - b2) * gW[k][i] ** 2
            W[k][i] -= (lr * (mW[k][i] / (1 - b1 ** t))) / (Math.sqrt(vW[k][i] / (1 - b2 ** t)) + eps)
          }
        for (let i = 0; i < N; i++) {
          mb[i] = b1 * mb[i] + (1 - b1) * gb[i]
          vb[i] = b2 * vb[i] + (1 - b2) * gb[i] ** 2
          b[i] -= (lr * (mb[i] / (1 - b1 ** t))) / (Math.sqrt(vb[i] / (1 - b2 ** t)) + eps)
        }
      }
      setState({ W: W.map((r) => [...r]), b: [...b], loss, step })
      if (step < STEPS) raf.current = requestAnimationFrame(tick)
    }
    raf.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf.current)
  }, [density, decay, seed])
  return state
}

const COLORS = ['#7c3aed', '#2563eb', '#0891b2', '#16a34a', '#d97706']

export default function ToySuperposition(_: VizProps) {
  const [logS, setLogS] = useState(-1) // density = 10^logS
  const [decay, setDecay] = useState(0.8)
  const [seed, setSeed] = useState(1)
  const density = 10 ** logS
  const st = useToyModel(density, decay, seed)
  if (!st) return <p>Training…</p>
  const cols = [...Array(N).keys()].map((i) => [st.W[0][i], st.W[1][i]])
  const norms = cols.map(([a, c]) => Math.hypot(a, c))
  // Feature dimensionality D_i = ||W_i||² / Σ_j (Ŵ_i · W_j)²  (Elhage et al., "Feature dimensionality").
  const dims = cols.map((wi, i) => {
    if (norms[i] < 1e-6) return 0
    const hat = [wi[0] / norms[i], wi[1] / norms[i]]
    const denom = cols.reduce((acc, wj) => acc + (hat[0] * wj[0] + hat[1] * wj[1]) ** 2, 0)
    return norms[i] ** 2 / denom
  })
  const S = 220
  const scale = 80 / Math.max(1, ...norms)
  return (
    <div>
      <HowTo>
        Five features (colored, more important = purple end) must share a <b>2-dimensional</b> hidden space. Each arrow is a
        feature's direction W<sub>i</sub>. Slide toward rare features: the model stops dropping the unimportant ones and packs
        all five into the plane as a pentagon — superposition. It is training live in your browser.
      </HowTo>
      <div className="viz-controls">
        <label>
          feature probability 1 − S = {density.toFixed(3)}
          <input type="range" min={-2} max={0} step={0.05} value={logS} onChange={(e) => setLogS(+e.target.value)} />
        </label>
        <label>
          importance decay {decay.toFixed(2)}
          <input type="range" min={0.5} max={1} step={0.05} value={decay} onChange={(e) => setDecay(+e.target.value)} />
        </label>
        <button className="chip" onClick={() => setSeed((s) => s + 1)}>
          re-initialize
        </button>
      </div>
      <div className="two-col">
        <svg viewBox={`0 0 ${S} ${S}`} style={{ width: '100%', maxWidth: 360 }} role="img" aria-label="feature directions">
          <defs>
            {COLORS.map((c, i) => (
              <marker key={i} id={`tm-${i}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
                <path d="M0,0 L10,5 L0,10 z" fill={c} />
              </marker>
            ))}
          </defs>
          <circle cx={S / 2} cy={S / 2} r={80} fill="none" stroke="currentColor" opacity={0.15} />
          <line x1={10} x2={S - 10} y1={S / 2} y2={S / 2} stroke="currentColor" opacity={0.1} />
          <line y1={10} y2={S - 10} x1={S / 2} x2={S / 2} stroke="currentColor" opacity={0.1} />
          {cols.map(([a, c], i) => (
            <line key={i} x1={S / 2} y1={S / 2} x2={S / 2 + a * scale} y2={S / 2 - c * scale} stroke={COLORS[i]} strokeWidth={3} markerEnd={`url(#tm-${i})`} opacity={norms[i] > 0.05 ? 1 : 0.25} />
          ))}
        </svg>
        <div>
          <Bars bars={dims.map((v, i) => ({ label: `feature ${i + 1} (importance ${(decay ** i).toFixed(2)})`, value: v, color: COLORS[i] }))} xLabel="dimensions per feature Dᵢ (1 = own axis, ⅖ = pentagon, 0 = dropped)" domain={[0, 1]} />
          <Kpis
            items={[
              ['training step', `${st.step} / ${STEPS}`],
              ['loss', st.loss.toFixed(4)],
              ['features represented (‖Wᵢ‖ > 0.5)', norms.filter((v) => v > 0.5).length],
            ]}
          />
        </div>
      </div>
    </div>
  )
}
