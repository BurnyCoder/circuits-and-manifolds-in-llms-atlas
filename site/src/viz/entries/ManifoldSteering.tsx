// Manifold steering: for a pair of days, the model's 7-day output distribution along the manifold path vs the
// straight chord (stacked bars per path point), plus energy and "visits intermediate days" metrics per layer.
import * as d3 from 'd3'
import { useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import { HowTo } from '../primitives/common'

interface Ex {
  from: number
  to: number
  manifold: number[][]
  linear: number[][]
}
interface Res {
  layer: number
  accuracy: number
  E_manifold: number
  E_linear: number
  via_manifold: number
  via_linear: number
  n_paths: number
  examples: Ex[]
}
interface D {
  layers: Res[]
  days: string[]
}

function Stack({ P, days }: { P: number[][]; days: string[] }) {
  const W = 420
  const H = 160
  const bw = W / P.length
  return (
    <svg viewBox={`0 0 ${W} ${H + 18}`} style={{ width: '100%', maxWidth: W }}>
      {P.map((p, i) => {
        let y = H
        return p.map((v, k) => {
          const h = v * H
          y -= h
          return <rect key={`${i}-${k}`} x={i * bw + 1} y={y} width={bw - 2} height={h} fill={d3.interpolateRainbow(k / 7)}>
            <title>
              step {i}: {days[k].trim()} {(v * 100).toFixed(0)}%
            </title>
          </rect>
        })
      })}
      <text x={0} y={H + 14} fontSize={11} fill="currentColor">
        start
      </text>
      <text x={W} y={H + 14} fontSize={11} fill="currentColor" textAnchor="end">
        target
      </text>
    </svg>
  )
}

export default function ManifoldSteering({ data }: VizProps) {
  const d = data as unknown as D
  const [li, setLi] = useState(0)
  const L = d.layers[li]
  const [ei, setEi] = useState(0)
  const ex = L.examples[ei]
  return (
    <div>
      <HowTo>
        Qwen3-1.7B answers “Let's do some day of the week math…”. We fit a smooth closed curve through the activations of the
        seven answers and push the model's state from one day to another, either <b>along the curve</b> or along the straight
        line between them. Each bar is the model's predicted day at one step (colors = days). Along the curve the answer should
        walk through the days in between; along the line it tends to jump.
      </HowTo>
      <div className="viz-controls">
        layer
        <select className="sel" value={li} onChange={(e) => setLi(+e.target.value)}>
          {d.layers.map((x, i) => (
            <option key={x.layer} value={i}>
              {x.layer}
            </option>
          ))}
        </select>
        {L.examples.map((x, i) => (
          <button key={i} className={`chip ${i === ei ? 'on' : ''}`} onClick={() => setEi(i)}>
            {d.days[x.from].trim()} → {d.days[x.to].trim()}
          </button>
        ))}
      </div>
      {ex && (
        <div className="two-col">
          <div>
            <h3>along the manifold</h3>
            <Stack P={ex.manifold} days={d.days} />
          </div>
          <div>
            <h3>straight line</h3>
            <Stack P={ex.linear} days={d.days} />
          </div>
        </div>
      )}
      <div className="viz-controls">
        {d.days.map((x, k) => (
          <span key={x} className="tag" style={{ background: d3.interpolateRainbow(k / 7), color: '#000' }}>
            {x.trim()}
          </span>
        ))}
      </div>
      <Bars
        bars={d.layers.flatMap((r) => [
          { label: `L${r.layer} · off-manifold energy, manifold path`, value: r.E_manifold, color: 'var(--c-manifold)' },
          { label: `L${r.layer} · off-manifold energy, straight line`, value: r.E_linear, color: '#94a3b8' },
        ])}
        xLabel="mean Bhattacharyya distance to the behaviour manifold (lower = more natural outputs)"
      />
    </div>
  )
}
