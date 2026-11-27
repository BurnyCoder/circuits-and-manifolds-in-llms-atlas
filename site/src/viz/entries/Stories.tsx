// Stories in space: pick a story, scrub through its sentences, see the model's six emotion ratings and the story's
// trajectory in a valence–arousal plane; probe R² by layer.
import { useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import LineChart from '../primitives/LineChart'
import { HowTo } from '../primitives/common'

interface D {
  emotions: string[]
  va: [number, number][]
  probe: { layer: number; r2: number }[]
  trajectories: { sentences: string[]; ratings: number[][]; xy: [number, number][] }[]
}
const EC = ['#16a34a', '#2563eb', '#dc2626', '#d97706', '#7c3aed', '#65a30d']

export default function Stories({ data }: VizProps) {
  const d = data as unknown as D
  const [si, setSi] = useState(0)
  const st = d.trajectories[si]
  const [t, setT] = useState(st.sentences.length - 1)
  const tt = Math.min(t, st.sentences.length - 1)
  const W = 360
  const xs = (v: number) => 30 + ((v - 1) / 8) * (W - 60)
  const ys = (a: number) => W - 30 - ((a - 1) / 8) * (W - 60)
  return (
    <div>
      <HowTo>
        The model reads a children's story one sentence at a time; after each sentence we ask how happy, sad, angry, surprised,
        afraid or disgusted the story feels so far. The path on the right places those ratings in a valence (pleasant →) ×
        arousal (calm → excited ↑) plane using human ratings of the six emotion words.
      </HowTo>
      <div className="viz-controls">
        <select className="sel" value={si} onChange={(e) => (setSi(+e.target.value), setT(99))}>
          {d.trajectories.map((x, i) => (
            <option key={i} value={i}>
              story {i + 1}: {x.sentences[0].slice(0, 50)}…
            </option>
          ))}
        </select>
        <label>
          sentence {tt + 1}/{st.sentences.length}
          <input type="range" min={0} max={st.sentences.length - 1} value={tt} onChange={(e) => setT(+e.target.value)} />
        </label>
      </div>
      <p style={{ fontSize: 14 }}>
        {st.sentences.map((s, i) => (
          <span key={i} style={{ opacity: i <= tt ? 1 : 0.3, fontWeight: i === tt ? 700 : 400 }}>
            {s}{' '}
          </span>
        ))}
      </p>
      <div className="two-col">
        <Bars bars={d.emotions.map((e, i) => ({ label: e, value: st.ratings[tt][i], color: EC[i] }))} xLabel="model's rating (0–1)" domain={[0, 1]} />
        <svg viewBox={`0 0 ${W} ${W}`} style={{ width: '100%', maxWidth: W }}>
          <rect x={30} y={30} width={W - 60} height={W - 60} fill="none" stroke="currentColor" opacity={0.2} />
          <text x={W / 2} y={W - 8} textAnchor="middle" fontSize={11} fill="currentColor">
            valence →
          </text>
          <text x={12} y={W / 2} textAnchor="middle" fontSize={11} fill="currentColor" transform={`rotate(-90 12 ${W / 2})`}>
            arousal →
          </text>
          {d.va.map(([v, a], i) => (
            <text key={i} x={xs(v)} y={ys(a)} fontSize={10} fill={EC[i]} textAnchor="middle">
              {d.emotions[i]}
            </text>
          ))}
          <polyline points={st.xy.slice(0, tt + 1).map(([v, a]) => `${xs(v)},${ys(a)}`).join(' ')} fill="none" stroke="currentColor" strokeWidth={2} />
          {st.xy.slice(0, tt + 1).map(([v, a], i) => (
            <circle key={i} cx={xs(v)} cy={ys(a)} r={i === tt ? 6 : 3} fill={i === tt ? '#f59e0b' : 'currentColor'} />
          ))}
        </svg>
      </div>
      <LineChart series={[{ name: 'probe R² (held-out stories)', x: d.probe.map((p) => p.layer), y: d.probe.map((p) => p.r2), color: 'var(--c-manifold)' }]} xLabel="layer" yLabel="R²" fmtX={(v) => String(v)} height={200} />
    </div>
  )
}
