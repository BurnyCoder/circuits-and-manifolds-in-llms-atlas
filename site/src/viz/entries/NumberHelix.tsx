// Number helix: 3-D helix (linear axis × a circle of chosen period), per-period circles, helix fit by layer vs
// controls, the embedding Fourier spectrum, activation patching, and scaling across the Pythia family.
import * as d3 from 'd3'
import { useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import LineChart from '../primitives/LineChart'
import Plot3D from '../primitives/Plot3D'
import Scatter2D from '../primitives/Scatter2D'
import { HowTo, Kpis, Tabs } from '../primitives/common'

type Fit = { in: number; test: number }
interface M {
  name: string
  n_layers: number
  spectrum: { period: number[]; mag: number[] }
  layers: ({ layer: number } & Record<string, Fit | number>)[]
  best_layer: number
  circles: Record<string, [number, number][]>
  linear: number[]
  accuracy: number
  n_pairs_correct: number
  patch: ({ layer: number; n: number } & Record<string, number | null>)[]
}
interface D {
  models: M[]
  periods: number[]
}

const BASES = ['helix', 'circles only (no linear term)', 'linear only', 'shuffled helix (control)']
const BASE_COLORS = ['var(--c-manifold)', '#8b5cf6', '#f59e0b', '#94a3b8']
const PATCHES = ['full layer', 'helix', 'circles only (no linear term)', 'linear only', 'PCA-9']
const short = (n: string) => n.split('/').pop()!

export default function NumberHelix({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'helix' | 'circles' | 'fit' | 'spectrum' | 'patch' | 'scale'>('helix')
  const [mi, setMi] = useState(d.models.length - 1)
  const [T, setT] = useState('10')
  const m = d.models[mi]
  const a = [...Array(100).keys()]
  const color = (i: number, period: number) => d3.interpolateRainbow((i % period) / period)
  const selector = (
    <div className="viz-controls">
      model
      <select className="sel" value={mi} onChange={(e) => setMi(+e.target.value)}>
        {d.models.map((x, i) => (
          <option key={x.name} value={i}>
            {short(x.name)}
          </option>
        ))}
      </select>
      <span className="viz-note">best layer {m.best_layer} of {m.n_layers}</span>
    </div>
  )
  return (
    <div>
      <Tabs
        tabs={[
          ['helix', '1 · The helix'],
          ['circles', '2 · Circles for T = 2, 5, 10, 100'],
          ['fit', '3 · Fit vs controls by layer'],
          ['spectrum', '4 · Fourier spectrum'],
          ['patch', '5 · Patching'],
          ['scale', '6 · Across model sizes'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'helix' && (
        <>
          <HowTo>
            The residual stream at the token of a number a (0–99), projected on the directions our helix fit assigns to “a”
            (vertical) and to cos/sin(2πa/T). Numbers wind around like a spring: one turn every T numbers. Drag to rotate; pick
            T.
          </HowTo>
          {selector}
          <div className="viz-controls">
            {d.periods.map((p) => (
              <label key={p}>
                <input type="radio" checked={T === String(p)} onChange={() => setT(String(p))} /> T = {p}
              </label>
            ))}
          </div>
          <Plot3D
            traces={[
              { x: m.circles[T].map((p) => p[0]), y: m.circles[T].map((p) => p[1]), z: m.linear, mode: 'lines', color: '#94a3b8', width: 2 },
              { x: m.circles[T].map((p) => p[0]), y: m.circles[T].map((p) => p[1]), z: m.linear, color: a.map((i) => d3.interpolateViridis(i / 99)), size: 4, text: a.map((i) => `a = ${i}`) },
            ]}
            axes={[`cos(2πa/${T})`, `sin(2πa/${T})`, 'linear term (∝ a)']}
          />
        </>
      )}
      {tab === 'circles' && (
        <>
          <HowTo>Each panel: numbers 0–99 projected on one period's cos/sin directions, colored by a mod T. A clean circle means that period is represented.</HowTo>
          {selector}
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 220px), 1fr))' }}>
            {d.periods.map((p) => (
              <figure key={p} style={{ margin: 0 }}>
                <Scatter2D xy={m.circles[String(p)]} colors={a.map((i) => color(i, p))} size={4} width={240} height={240} />
                <figcaption className="viz-note" style={{ textAlign: 'center' }}>
                  T = {p} (color = a mod {p})
                </figcaption>
              </figure>
            ))}
          </div>
        </>
      )}
      {tab === 'fit' && (
        <>
          <HowTo>
            Held-out R²: fit the basis on 80 numbers, predict the activations of the other 20. The shuffled helix uses the same
            basis with numbers in random order (control).
          </HowTo>
          {selector}
          <LineChart
            series={BASES.map((b, i) => ({ name: b, x: m.layers.map((l) => l.layer), y: m.layers.map((l) => (l[b] as Fit).test), color: BASE_COLORS[i], dash: i === 3 ? '5 4' : undefined }))}
            xLabel="layer (0 = embeddings)"
            yLabel="held-out R²"
            fmtX={(v) => String(v)}
          />
        </>
      )}
      {tab === 'spectrum' && (
        <>
          <HowTo>Fourier transform of the number-token embeddings over a = 0…99. Peaks mark periods the model uses; dashed lines at T = 2, 5, 10.</HowTo>
          {selector}
          <LineChart
            series={[{ name: 'embedding Fourier magnitude', x: m.spectrum.period, y: m.spectrum.mag, color: 'var(--c-manifold)' }]}
            xLog
            xLabel="period T (log scale)"
            yLabel="magnitude"
            fmtX={(v) => String(v)}
            bands={[2, 5, 10].map((p) => ({ x0: p * 0.97, x1: p * 1.03, label: `T=${p}`, color: '#f59e0b' }))}
          />
        </>
      )}
      {tab === 'patch' && (
        <>
          <HowTo>
            Run “a′+b=” (wrong a), then move the activation at the a′ token toward a — fully (full layer) or only along a fitted
            subspace — and see what fraction of the clean logit difference is recovered (median over {m.patch[0]?.n ?? 0}{' '}
            prompts the model answers correctly).
          </HowTo>
          {selector}
          <Bars
            bars={m.patch.flatMap((r) =>
              PATCHES.map((p, i) => ({ label: `L${r.layer} · ${p}`, value: (r[p] as number) ?? 0, color: ['#64748b', 'var(--c-manifold)', '#8b5cf6', '#f59e0b', '#ec4899'][i] })),
            )}
            xLabel="fraction of the clean logit difference recovered"
          />
          <Kpis items={[['addition accuracy (a + b < 100, few-shot)', `${(m.accuracy * 100).toFixed(0)}%`]]} />
        </>
      )}
      {tab === 'scale' && (
        <>
          <HowTo>Does the helix get cleaner with scale? Best-layer held-out R² and addition accuracy across the Pythia models we ran.</HowTo>
          <Bars bars={d.models.map((x) => ({ label: short(x.name), value: (x.layers[x.best_layer].helix as Fit).test, color: 'var(--c-manifold)' }))} xLabel="best-layer held-out helix R²" domain={[0, 1]} />
          <Bars bars={d.models.map((x) => ({ label: short(x.name), value: x.accuracy, color: '#f59e0b' }))} xLabel="few-shot addition accuracy" domain={[0, 1]} />
        </>
      )}
    </div>
  )
}
