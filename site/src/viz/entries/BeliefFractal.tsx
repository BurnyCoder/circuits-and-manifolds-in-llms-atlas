// Belief-state fractal: ground-truth Bayesian beliefs over Mess3's hidden states (a fractal in the triangle) next to
// the model's linear readout of them, scrubbable across training; loss curve vs the optimal predictor.
import { useState } from 'react'
import type { VizProps } from '../registry'
import LineChart from '../primitives/LineChart'
import Scatter2D from '../primitives/Scatter2D'
import { HowTo, Kpis } from '../primitives/common'

interface D {
  truth_xy: [number, number][]
  truth_rgb: [number, number, number][]
  snaps: { step: number; r2: number; mse: number; xy: [number, number][] }[]
  rand_xy: [number, number][]
  loss_curve: number[]
  loss_every: number
  summary: Record<string, number>
}

const TRI: [number, number][] = [
  [0, 0],
  [1, 0],
  [0.5, Math.sqrt(3) / 2],
]

export default function BeliefFractal({ data }: VizProps) {
  const d = data as unknown as D
  const [si, setSi] = useState(d.snaps.length - 1)
  const colors = d.truth_rgb.map(([r, g, b]) => `rgb(${r},${g},${b})`)
  const s = d.snaps[si]
  const bounds: [number, number, number, number] = [-0.05, 1.05, -0.08, 0.95]
  return (
    <div>
      <HowTo>
        Mess3 is a tiny hidden Markov process with 3 hidden states. After each token, the ideal (Bayesian) observer holds a
        belief over the states — a point in a triangle. Over all histories these beliefs form a <b>fractal</b> (left). A
        4-layer transformer trained only to predict the next token ends up with the same fractal, linearly, in its residual
        stream (right; same colors = same true belief). Drag through training.
      </HowTo>
      <div className="viz-controls">
        <label>
          training step {s.step.toLocaleString()}
          <input type="range" min={0} max={d.snaps.length - 1} value={si} onChange={(e) => setSi(+e.target.value)} />
        </label>
      </div>
      <div className="two-col">
        <figure style={{ margin: 0 }}>
          <Scatter2D xy={d.truth_xy} colors={colors} size={1.6} bounds={bounds} polygon={TRI} alpha={0.9} />
          <figcaption className="viz-note">Ground truth: Bayesian belief states</figcaption>
        </figure>
        <figure style={{ margin: 0 }}>
          <Scatter2D xy={s.xy} colors={colors} size={1.6} bounds={bounds} polygon={TRI} alpha={0.9} />
          <figcaption className="viz-note">
            Linear readout of the residual stream at step {s.step.toLocaleString()} (held-out R² {s.r2.toFixed(3)})
          </figcaption>
        </figure>
      </div>
      <Kpis
        items={[
          ['belief R² at this step', s.r2.toFixed(3)],
          ['random-init network (control)', d.summary.rand_r2.toFixed(3)],
          ['final loss / optimal / uniform (nats)', `${d.summary.final_loss.toFixed(3)} / ${d.summary.optimal_loss.toFixed(3)} / ${d.summary.uniform_loss.toFixed(3)}`],
        ]}
      />
      <LineChart
        series={[
          { name: 'training loss', x: d.loss_curve.map((_, i) => i * d.loss_every + 1), y: d.loss_curve, color: 'var(--c-manifold)' },
          { name: 'optimal (entropy rate)', x: [1, d.loss_curve.length * d.loss_every], y: [d.summary.optimal_loss, d.summary.optimal_loss], color: '#16a34a', dash: '5 4' },
        ]}
        xLog
        xLabel="training step (log scale)"
        yLabel="cross-entropy (nats)"
        height={220}
      />
    </div>
  )
}
