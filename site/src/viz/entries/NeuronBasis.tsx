// Neuron basis: loss/KL as more and more neuron activations are zeroed per token (keep top-k by |activation|),
// controls, and the SwiGLU sign-quadrant census vs. which quadrants carry causal attribution.
import { useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import Heatmap, { Legend } from '../primitives/Heatmap'
import LineChart from '../primitives/LineChart'
import { HowTo, Kpis, Tabs, seqScale } from '../primitives/common'

interface D {
  curve: { k: number; frac_ablated: number; loss: number; kl: number }[]
  controls: Record<string, { loss: number; kl: number }>
  census: number[][]
  attr_frac: number[]
  mag_frac: number[]
  all_frac: number[]
  summary: Record<string, number>
}
const Q = ['Q1 gate+ up+', 'Q2 gate+ up−', 'Q3 gate− up−', 'Q4 gate− up+']
const QC = ['#16a34a', '#2563eb', '#d97706', '#dc2626']

export default function NeuronBasis({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'sparsity' | 'quadrants'>('sparsity')
  const [ki, setKi] = useState(d.curve.findIndex((c) => c.k === 480))
  const c = d.curve[Math.max(0, ki)]
  const base = d.curve[0].loss
  return (
    <div>
      <Tabs
        tabs={[
          ['sparsity', '1 · How many neurons does each token need?'],
          ['quadrants', '2 · SwiGLU sign quadrants'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'sparsity' && (
        <>
          <HowTo>
            For every token, in every one of Qwen3-1.7B's 28 MLP layers, keep only the k neurons with the largest |activation|
            out of 6,144 and zero the rest. Slide k down: the model barely notices until very few neurons are left. Controls:
            keep a random k instead, or zero only the top 480.
          </HowTo>
          <div className="viz-controls">
            <label>
              keep k = {c.k} neurons per layer ({(c.frac_ablated * 100).toFixed(1)}% zeroed)
              <input type="range" min={0} max={d.curve.length - 1} value={ki} onChange={(e) => setKi(+e.target.value)} />
            </label>
          </div>
          <LineChart
            series={[
              { name: 'loss (nats)', x: d.curve.map((x) => x.k), y: d.curve.map((x) => x.loss), color: 'var(--c-neuron)', width: 3 },
              { name: 'unablated loss', x: [d.curve[d.curve.length - 1].k, 6144], y: [base, base], color: '#94a3b8', dash: '5 4' },
            ]}
            xLog
            xLabel="neurons kept per token per layer (log scale, of 6,144)"
            yLabel="next-token loss (nats)"
            marker={c.k}
            fmtX={(v) => String(v)}
          />
          <Kpis
            items={[
              ['loss at this k', `${c.loss.toFixed(3)} (+${((c.loss / base - 1) * 100).toFixed(1)}%)`],
              ['KL from the full model', c.kl.toFixed(3)],
              ['random 480 kept (control)', d.controls.random.loss.toFixed(3)],
              ['only the top 480 zeroed', d.controls.drop_topk.loss.toFixed(3)],
              ['P25–P75 band mean-ablated', d.controls.band_p25_p75.loss.toFixed(3)],
            ]}
          />
        </>
      )}
      {tab === 'quadrants' && (
        <>
          <HowTo>
            A SwiGLU neuron outputs SiLU(gate)·up. The signs of gate and up split activations into four quadrants. Most
            activations are “gate-closed” (Q3/Q4: small, since SiLU of a negative number is near zero), but the cells that
            matter most for the loss (gradient × activation, top 1%) are gate-open.
          </HowTo>
          <div className="two-col">
            <div>
              <h3>All activations</h3>
              <Bars bars={Q.map((q, i) => ({ label: q, value: d.all_frac[i], color: QC[i] }))} xLabel="share of activations" domain={[0, 1]} />
              <h3>Top-1% attributed cells</h3>
              <Bars bars={Q.map((q, i) => ({ label: q, value: d.attr_frac[i], color: QC[i] }))} xLabel="share of most important cells" domain={[0, 1]} />
            </div>
            <div>
              <h3>Quadrant share by layer</h3>
              <Heatmap values={d.census} color={seqScale(0.6)} cell={22} xLabels={['Q1', 'Q2', 'Q3', 'Q4']} yLabels={d.census.map((_, i) => `L${i}`)} yTitle="layer" maxLabelEvery={2} tooltip={(r, col, v) => `layer ${r}, ${Q[col]}: ${(v * 100).toFixed(1)}%`} />
              <Legend color={seqScale(0.6)} lo={0} hi={0.6} label="share of activations" />
            </div>
          </div>
        </>
      )}
    </div>
  )
}
