// Linebreaks in Claude 3.5 Haiku (Gurnee et al. 2025) — three live simulations of the published mechanism (not model
// data): (1) place-cell-like count features with dilating receptive fields, (2) the paper's particle model of why the
// count manifold ripples (points on an n-sphere attract their neighbours and repel the rest; Fig. 13), (3) a boundary
// head's QK map rotating the count manifold onto the line-width manifold with an offset ε.
import * as d3 from 'd3'
import { useEffect, useMemo, useState } from 'react'
import type { VizProps } from '../registry'
import Heatmap, { Legend } from '../primitives/Heatmap'
import LineChart from '../primitives/LineChart'
import Plot3D from '../primitives/Plot3D'
import { HowTo, Kpis, Tabs, divScale } from '../primitives/common'

function pcaTop3(X: number[][]): number[][] {
  const n = X.length
  const d = X[0].length
  const mu = Array(d).fill(0)
  X.forEach((r) => r.forEach((v, j) => (mu[j] += v / n)))
  const C = X.map((r) => r.map((v, j) => v - mu[j]))
  const comps: number[][] = []
  for (let c = 0; c < 3; c++) {
    let v = Array.from({ length: d }, (_, j) => Math.sin(j * 1.7 + c * 3.1) + 0.1)
    for (let it = 0; it < 60; it++) {
      const w = Array(d).fill(0)
      C.forEach((r) => {
        const s = r.reduce((a, x, j) => a + x * v[j], 0)
        r.forEach((x, j) => (w[j] += s * x))
      })
      comps.forEach((u) => {
        const p = w.reduce((a, x, j) => a + x * u[j], 0)
        w.forEach((_, j) => (w[j] -= p * u[j]))
      })
      const nrm = Math.hypot(...w) || 1
      v = w.map((x) => x / nrm)
    }
    comps.push(v)
  }
  return C.map((r) => comps.map((u) => r.reduce((a, x, j) => a + x * u[j], 0)))
}

function useParticles(n: number, dim: number, zone: number, circle: boolean) {
  const [X, setX] = useState<number[][]>([])
  useEffect(() => {
    let raf = 0
    let seed = 7
    const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 2 - 1
    let P = Array.from({ length: n }, () => {
      const v = Array.from({ length: dim }, rnd)
      const s = Math.hypot(...v)
      return v.map((x) => x / s)
    })
    let it = 0
    const step = () => {
      for (let k = 0; k < 20; k++, it++) {
        // Gradient descent on L = Σ_{i≠j} (x_i·x_j − K_ij)², with K_ij = 1 for counts within ±zone, else 0: nearby counts
        // should be similar and distant ones orthogonal (Gurnee et al. 2025, "optimal packing" argument). In few
        // dimensions the best fit keeps only low Fourier modes of this box kernel, so similarities overshoot below zero
        // and come back ("ringing", the Gibbs phenomenon) and the curve ripples.
        const F = P.map(() => Array(dim).fill(0))
        for (let i = 0; i < n; i++)
          for (let j = 0; j < n; j++) {
            if (i === j) continue
            const gap = circle ? Math.min(Math.abs(i - j), n - Math.abs(i - j)) : Math.abs(i - j)
            let dot = 0
            for (let t = 0; t < dim; t++) dot += P[i][t] * P[j][t]
            const r = dot - (gap <= zone ? 1 : 0) // residual of this pair's similarity
            for (let t = 0; t < dim; t++) F[i][t] -= (4 * r * P[j][t]) / n // −∂L/∂x_i
          }
        P = P.map((p, i) => {
          const q = p.map((x, t) => x + 0.02 * F[i][t])
          const s = Math.hypot(...q)
          return q.map((x) => x / s) // stay on the unit sphere
        })
      }
      setX(P.map((p) => [...p]))
      if (it < 3000) raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [n, dim, zone, circle])
  return X
}

export default function LinebreaksClaude(_: VizProps) {
  const [tab, setTab] = useState<'cells' | 'ripple' | 'qk'>('ripple')
  const [count, setCount] = useState(40)
  const [dim, setDim] = useState(6)
  const [zone, setZone] = useState(6)
  const [circle, setCircle] = useState(false)
  const [eps, setEps] = useState(8)
  const X = useParticles(60, dim, zone, circle)
  const P3 = useMemo(() => (X.length ? pcaTop3(X) : []), [X])
  const cos = useMemo(() => X.map((a) => X.map((b) => a.reduce((s, v, t) => s + v * b[t], 0))), [X])
  // Place cells: 10 features whose centres and widths grow with count (dilating receptive fields).
  const centres = [...Array(10).keys()].map((i) => 6 + 150 * ((i / 9) ** 1.5) * 0.95)
  const width = (c: number) => 4 + c * 0.12
  const xs = [...Array(150).keys()].map((i) => i + 1)
  const cell = (k: number, x: number) => Math.exp(-((x - centres[k]) ** 2) / (2 * width(centres[k]) ** 2))
  const qkColor = useMemo(() => divScale(1), [])
  const N = 40
  const theta = (i: number) => (2 * Math.PI * i) / 60
  const qk = [...Array(N).keys()].map((k) => [...Array(N).keys()].map((i) => Math.cos(theta(i) + theta(eps) - theta(k))))
  return (
    <div>
      <Tabs
        tabs={[
          ['cells', '1 · Count features like place cells'],
          ['ripple', '2 · Why the manifold ripples (simulation)'],
          ['qk', '3 · Boundary heads rotate the manifold'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'cells' && (
        <>
          <HowTo>Sketch of the published finding: about ten sparse features each respond to a range of character counts; ranges widen at larger counts, and 2–3 overlap at any moment (illustrative curves).</HowTo>
          <div className="viz-controls">
            <label>
              characters since newline: {count}
              <input type="range" min={1} max={150} value={count} onChange={(e) => setCount(+e.target.value)} />
            </label>
          </div>
          <LineChart series={centres.map((_, k) => ({ name: `feature ${k + 1}`, x: xs, y: xs.map((x) => cell(k, x)), color: d3.interpolateTurbo(k / 9), width: cell(k, count) > 0.3 ? 3 : 1.2 }))} xLabel="characters since the last newline" yLabel="activation (illustrative)" marker={count} fmtX={(v) => String(v)} height={260} />
        </>
      )}
      {tab === 'ripple' && (
        <>
          <HowTo>
            The paper's packing argument, running live: 60 points (counts) on a sphere in {dim} dimensions try to be similar to
            counts within ±{zone} and orthogonal to all others. With few dimensions this cannot be met exactly; the best fit is a
            curve that <b>ripples</b>, and its similarity matrix “rings” (similarity dips below zero, then rises again), as in
            Claude. Fewer dimensions → stronger ringing.
          </HowTo>
          <div className="viz-controls">
            <label>
              dimensions {dim}
              <input type="range" min={3} max={8} value={dim} onChange={(e) => setDim(+e.target.value)} />
            </label>
            <label>
              neighbour zone ±{zone}
              <input type="range" min={2} max={15} value={zone} onChange={(e) => setZone(+e.target.value)} />
            </label>
            <label>
              <input type="checkbox" checked={circle} onChange={(e) => setCircle(e.target.checked)} /> circle (wrap-around) instead of interval
            </label>
          </div>
          <div className="two-col">
            {P3.length > 0 && <Plot3D traces={[{ x: P3.map((p) => p[0]), y: P3.map((p) => p[1]), z: P3.map((p) => p[2]), mode: 'lines+markers', color: P3.map((_, i) => d3.interpolateTurbo(i / 59)), size: 3 }]} height={320} />}
            {cos.length > 0 && (
              <div>
                <Heatmap values={cos} color={qkColor} cell={5} xTitle="count" yTitle="count" tooltip={(r, c, v) => `${r} vs ${c}: ${v.toFixed(2)}`} />
                <Legend color={qkColor} lo={-1} hi={1} label="cosine similarity" />
              </div>
            )}
          </div>
        </>
      )}
      {tab === 'qk' && (
        <>
          <HowTo>
            A boundary head compares “characters so far” (query) with “line width” (key). Its QK matrix rotates the count manifold so
            that count i lines up with width i + ε — the head fires as the line approaches its end. Drag ε (schematic: counts as angles
            on a circle).
          </HowTo>
          <div className="viz-controls">
            <label>
              offset ε = {eps}
              <input type="range" min={0} max={20} value={eps} onChange={(e) => setEps(+e.target.value)} />
            </label>
          </div>
          <Heatmap values={qk} color={qkColor} cell={8} xTitle="characters so far (query)" yTitle="line width (key)" tooltip={(r, c, v) => `width ${r}, count ${c}: ${v.toFixed(2)}`} />
          <Kpis
            items={[
              ['count ↔ width probe similarity, residual stream', '≈ 0.25'],
              ['… after the QK map', '≈ 1'],
              ['heads per set, different offsets', '≈ 3'],
              ['characters-remaining subspace (2 PCs)', '92%'],
              ['newline decision AUC', '0.91'],
            ]}
          />
          <p className="viz-note">Published values for Claude 3.5 Haiku (Gurnee et al. 2025, §3–4).</p>
        </>
      )}
    </div>
  )
}
