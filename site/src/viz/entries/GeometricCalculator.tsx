// Geometric calculator: (1) an interactive base-10 "clock" simulation of Feucht et al.'s mechanism (month → number →
// add on circles of period 2, 5, 10 → back to month), (2) our Fourier-probe spectra on "{a}+{b}=" in Pythia models.
import * as d3 from 'd3'
import { useState } from 'react'
import type { VizProps } from '../registry'
import LineChart from '../primitives/LineChart'
import { HowTo, Tabs } from '../primitives/common'

interface D {
  models: { name: string; layers: { layer: number; r2: number[]; top: number[] }[]; best_layer: number }[]
  periods: number[]
  base10: number[]
}
const MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

function Clock({ T, value, color }: { T: number; value: number; color: string }) {
  const S = 120
  const a = (2 * Math.PI * (value % T)) / T - Math.PI / 2
  return (
    <svg viewBox={`0 0 ${S} ${S}`} style={{ width: 120 }}>
      <circle cx={60} cy={60} r={48} fill="none" stroke="currentColor" opacity={0.3} />
      {[...Array(T).keys()].map((k) => {
        const b = (2 * Math.PI * k) / T - Math.PI / 2
        return <circle key={k} cx={60 + 48 * Math.cos(b)} cy={60 + 48 * Math.sin(b)} r={T > 20 ? 1 : 3} fill="currentColor" opacity={0.5} />
      })}
      <line x1={60} y1={60} x2={60 + 44 * Math.cos(a)} y2={60 + 44 * Math.sin(a)} stroke={color} strokeWidth={4} strokeLinecap="round" />
      <text x={60} y={64} textAnchor="middle" fontSize={12} fill="currentColor">
        mod {T}
      </text>
      <text x={60} y={116} textAnchor="middle" fontSize={11} fill={color}>
        {value % T}
      </text>
    </svg>
  )
}

export default function GeometricCalculator({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'sim' | 'probes'>('sim')
  const [m, setM] = useState(7) // August
  const [k, setK] = useState(16)
  const [mi, setMi] = useState(d.models.length - 1)
  const sum = m + 1 + k // month number (1-12) + offset, computed in base 10
  const result = ((sum - 1) % 12) + 1
  const model = d.models[mi]
  const best = model.layers.find((l) => l.layer === model.best_layer) ?? model.layers[0]
  return (
    <div>
      <Tabs
        tabs={[
          ['sim', '1 · The base-10 clock (simulation)'],
          ['probes', '2 · Circles in Pythia (our probes)'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'sim' && (
        <>
          <HowTo>
            Feucht et al. found that Llama-3.1-8B answers “What month is 16 months after August?” not with a 12-hour clock but by
            turning August into the number 8, adding 16 in ordinary base 10 on circles of period 2, 5 and 10 (like the ones, even/odd
            and fives digits), and mapping 24 back to a month. This is an illustration of that published mechanism, not model data.
          </HowTo>
          <div className="viz-controls">
            <label>
              start month
              <select className="sel" value={m} onChange={(e) => setM(+e.target.value)}>
                {MONTHS.map((x, i) => (
                  <option key={x} value={i}>
                    {x}
                  </option>
                ))}
              </select>
            </label>
            <label>
              + {k} months
              <input type="range" min={1} max={24} value={k} onChange={(e) => setK(+e.target.value)} />
            </label>
          </div>
          <p style={{ fontSize: 18 }}>
            {MONTHS[m]} → <b>{m + 1}</b> &nbsp;+&nbsp; <b>{k}</b> &nbsp;=&nbsp; <b>{sum}</b> → {MONTHS[result - 1]}
          </p>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {[2, 5, 10, 100].map((T, i) => (
              <Clock key={T} T={T} value={sum} color={d3.schemeTableau10[i]} />
            ))}
          </div>
        </>
      )}
      {tab === 'probes' && (
        <>
          <HowTo>
            For each period T we train a linear probe to read sin/cos(2π(a+b)/T) from the last token of “a+b=” and report held-out R².
            Peaks at the dashed base-10 periods would mean the model stores the sum on those circles.
          </HowTo>
          <div className="viz-controls">
            <select className="sel" value={mi} onChange={(e) => setMi(+e.target.value)}>
              {d.models.map((x, i) => (
                <option key={x.name} value={i}>
                  {x.name.split('/').pop()} (best layer {x.best_layer})
                </option>
              ))}
            </select>
          </div>
          <LineChart
            series={model.layers.map((l, i) => ({ name: `layer ${l.layer}`, x: d.periods, y: l.r2, color: d3.schemeTableau10[i], width: l.layer === model.best_layer ? 3 : 1.2 }))}
            xLog
            xLabel="period T (log scale)"
            yLabel="probe R² (held out)"
            fmtX={(v) => String(v)}
            bands={d.base10.map((T) => ({ x0: T * 0.97, x1: T * 1.03, label: '', color: '#f59e0b' }))}
          />
          <p className="viz-note">Best layer's top periods: {best.top.join(', ')}</p>
        </>
      )}
    </div>
  )
}
