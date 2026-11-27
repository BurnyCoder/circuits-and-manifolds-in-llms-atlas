// Linebreak counting manifold (open models): 3-D curve of per-count mean activations, the cosine "ringing" matrix,
// probe R² by layer for Qwen3-1.7B vs GPT-2 small vs shuffled labels, and the model's newline behaviour.
import * as d3 from 'd3'
import { useMemo, useState } from 'react'
import type { VizProps } from '../registry'
import Heatmap, { Legend } from '../primitives/Heatmap'
import LineChart from '../primitives/LineChart'
import Plot3D from '../primitives/Plot3D'
import { HowTo, Kpis, Tabs, divScale } from '../primitives/common'

interface Geo {
  counts: number[]
  pcs: number[][]
  var: number[]
  cos: number[][]
  deg_per_char: number
}
interface R {
  name: string
  probe: { layer: number; r2: number; r2_shuffled: number }[]
  best_layer: number
  geo: Record<string, Geo>
  newline_auc: number
  n_tokens: number
}
interface D {
  qwen: R
  gpt2: R
}

export default function Linebreak({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'curve' | 'ring' | 'probe'>('curve')
  const layers = Object.keys(d.qwen.geo)
  const [L, setL] = useState(String(d.qwen.best_layer))
  const [pcSet, setPcSet] = useState<0 | 3>(0)
  const g = d.qwen.geo[L] ?? d.qwen.geo[layers[0]]
  const ringColor = useMemo(() => divScale(1), [])
  const cols = g.counts.map((c) => d3.interpolateTurbo(c / 150))
  return (
    <div>
      <Tabs
        tabs={[
          ['curve', '1 · The counting manifold'],
          ['ring', '2 · Ringing'],
          ['probe', '3 · Where is the count?'],
        ]}
        value={tab}
        onChange={setTab}
      />
      <div className="viz-controls">
        Qwen3-1.7B layer
        <select className="sel" value={L} onChange={(e) => setL(e.target.value)}>
          {layers.map((l) => (
            <option key={l} value={l}>
              {l}
              {Number(l) === d.qwen.best_layer ? ' (best probe)' : ''}
            </option>
          ))}
        </select>
      </div>
      {tab === 'curve' && (
        <>
          <HowTo>
            Average residual stream for every “characters since the last newline” value (1–150), projected on its top
            principal components. Instead of a straight ruler, the counts trace a <b>twisting curve</b> (color = count). Switch
            to PCs 4–6 to see the finer ripples.
          </HowTo>
          <div className="viz-controls">
            <label>
              <input type="radio" checked={pcSet === 0} onChange={() => setPcSet(0)} /> PCs 1–3
            </label>
            <label>
              <input type="radio" checked={pcSet === 3} onChange={() => setPcSet(3)} /> PCs 4–6
            </label>
          </div>
          <Plot3D
            traces={[
              { x: g.pcs.map((p) => p[pcSet]), y: g.pcs.map((p) => p[pcSet + 1]), z: g.pcs.map((p) => p[pcSet + 2]), mode: 'lines', color: '#94a3b8', width: 2 },
              { x: g.pcs.map((p) => p[pcSet]), y: g.pcs.map((p) => p[pcSet + 1]), z: g.pcs.map((p) => p[pcSet + 2]), color: cols, size: 3.5, text: g.counts.map((c) => `${c} chars`) },
            ]}
            axes={pcSet === 0 ? ['PC1', 'PC2', 'PC3'] : ['PC4', 'PC5', 'PC6']}
          />
          <Kpis
            items={[
              ['variance in 6 PCs', `${(g.var.reduce((a, b) => a + b, 0) * 100).toFixed(0)}%`],
              ['PC1–PC2 rotation', `${g.deg_per_char.toFixed(2)}° per character`],
              ['tokens analysed', d.qwen.n_tokens.toLocaleString()],
            ]}
          />
        </>
      )}
      {tab === 'ring' && (
        <>
          <HowTo>
            Cosine similarity between the mean vectors of every pair of counts. A diagonal band (neighbouring counts are similar)
            plus faint off-diagonal stripes is the “ringing” Anthropic found in Claude: the signature of a curve that ripples to
            pack many distinguishable positions into few dimensions.
          </HowTo>
          <Heatmap
            values={g.cos}
            color={ringColor}
            cell={Math.max(3, Math.floor(420 / g.cos.length))}
            xLabels={g.counts.filter((_, i) => i % 2 === 0).map(String)}
            yLabels={g.counts.filter((_, i) => i % 2 === 0).map(String)}
            xTitle="count"
            yTitle="count"
            maxLabelEvery={10}
            tooltip={(r, c, v) => `${g.counts[r * 2]} vs ${g.counts[c * 2]} chars: cos ${v.toFixed(2)}`}
          />
          <Legend color={ringColor} lo={-1} hi={1} label="cosine similarity" />
        </>
      )}
      {tab === 'probe' && (
        <>
          <HowTo>
            A linear (ridge) probe reads the character count from each layer, tested on held-out documents. Claude 3.5 Haiku
            had it almost perfectly after layer 1; compare the open models and a shuffled-label control.
          </HowTo>
          <LineChart
            series={[
              { name: 'Qwen3-1.7B', x: d.qwen.probe.map((p) => p.layer), y: d.qwen.probe.map((p) => p.r2), color: 'var(--c-manifold)', width: 3 },
              { name: 'GPT-2 small', x: d.gpt2.probe.map((p) => p.layer), y: d.gpt2.probe.map((p) => p.r2), color: '#f59e0b' },
              { name: 'Qwen3 shuffled labels', x: d.qwen.probe.map((p) => p.layer), y: d.qwen.probe.map((p) => p.r2_shuffled), color: '#94a3b8', dash: '5 4' },
            ]}
            xLabel="layer"
            yLabel="held-out R²"
            yDomain={[-0.1, 1]}
            fmtX={(v) => String(v)}
          />
          <Kpis
            items={[
              ['newline prediction AUC — Qwen3-1.7B', d.qwen.newline_auc.toFixed(3)],
              ['— GPT-2 small', d.gpt2.newline_auc.toFixed(3)],
            ]}
          />
        </>
      )}
    </div>
  )
}
