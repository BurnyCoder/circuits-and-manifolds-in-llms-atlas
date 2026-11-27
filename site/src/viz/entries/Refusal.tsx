// Refusal direction: harmful vs harmless prompts separating along one direction (per-layer PCA scrubber), refusal
// rates under ablation / addition vs random controls, and harmless examples where adding the direction causes refusal.
import { useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import LineChart from '../primitives/LineChart'
import Scatter2D from '../primitives/Scatter2D'
import { HowTo, Kpis, Tabs } from '../primitives/common'

interface D {
  rates: Record<string, number>
  selection: { layer: number; bypass: number; induce: number }[]
  base_logodds: { harmful: number; harmless: number }
  pcs: { layer: number; xy: [number, number][]; var: number[]; sep: number }[]
  examples: { prompt: string; baseline: string; steered: string }[]
  summary: Record<string, number>
}

export default function Refusal({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'geometry' | 'rates' | 'examples'>('rates')
  const [li, setLi] = useState(Math.floor(d.pcs.length / 2))
  const p = d.pcs[li]
  const half = p.xy.length / 2
  return (
    <div>
      <Tabs
        tabs={[
          ['rates', '1 · One direction controls refusal'],
          ['geometry', '2 · Harmful vs harmless, layer by layer'],
          ['examples', '3 · Refusing harmless requests'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'rates' && (
        <>
          <HowTo>
            The refusal direction is the difference between the average activation on harmful and harmless instructions (layer{' '}
            {d.summary.layer} of {d.summary.n_layers}). Projecting it out of every layer stops refusals; adding it makes the model
            refuse innocent requests. A random direction of the same size does neither. (Completions to harmful prompts are not
            stored; only whether the model refused.)
          </HowTo>
          <Bars
            bars={Object.entries(d.rates).map(([k, v]) => ({ label: k, value: v, color: k.includes('random') ? '#94a3b8' : k.includes('baseline') ? '#64748b' : 'var(--c-steering)' }))}
            xLabel={`refusal rate (substring judge, ${d.summary.n_test} test prompts each)`}
            domain={[0, 1]}
          />
          <LineChart
            series={[
              { name: 'harmful refusal log-odds with this layer’s direction ablated', x: d.selection.map((s) => s.layer), y: d.selection.map((s) => s.bypass), color: '#dc2626' },
              { name: 'harmless refusal log-odds with it added', x: d.selection.map((s) => s.layer), y: d.selection.map((s) => s.induce), color: 'var(--c-steering)' },
            ]}
            xLabel="candidate layer"
            yLabel="refusal log-odds (validation)"
            marker={d.summary.layer}
            fmtX={(v) => String(v)}
            height={240}
          />
        </>
      )}
      {tab === 'geometry' && (
        <>
          <HowTo>Last-token activations of harmful (red) and harmless (green) instructions, top-2 principal components per layer.</HowTo>
          <div className="viz-controls">
            <label>
              layer {p.layer}
              <input type="range" min={0} max={d.pcs.length - 1} value={li} onChange={(e) => setLi(+e.target.value)} />
            </label>
          </div>
          <Scatter2D xy={p.xy} colors={p.xy.map((_, i) => (i < half ? '#dc2626' : '#16a34a'))} size={4} alpha={0.75} />
          <Kpis items={[['class separation (mean distance / spread)', p.sep.toFixed(2)], ['variance in PC1–2', `${((p.var[0] + p.var[1]) * 100).toFixed(0)}%`]]} />
        </>
      )}
      {tab === 'examples' && (
        <>
          <HowTo>Harmless test prompts where the unsteered model helps and the model with the refusal direction added refuses.</HowTo>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>prompt</th>
                  <th>baseline</th>
                  <th>+ refusal direction</th>
                </tr>
              </thead>
              <tbody>
                {d.examples.map((e, i) => (
                  <tr key={i}>
                    <td>{e.prompt}</td>
                    <td>{e.baseline}</td>
                    <td>{e.steered}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
