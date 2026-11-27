// ActAdd: pick a steering vector and a coefficient; read real GPT-2-XL completions; on-topic / positive rates vs c.
import { useState } from 'react'
import type { VizProps } from '../registry'
import LineChart from '../primitives/LineChart'
import { HowTo } from '../primitives/common'

interface Row {
  coef: number
  kind: string
  wedding_rate: number
  positive_rate: number
  examples: { prompt: string; completion: string }[]
}
interface D {
  vectors: Record<string, { layer: number; plus: string; minus: string; norm: number; rows: Row[] }>
  coefs: number[]
}

export default function ActAdd({ data }: VizProps) {
  const d = data as unknown as D
  const names = Object.keys(d.vectors)
  const [name, setName] = useState(names[0])
  const v = d.vectors[name]
  const coefs = [...new Set(v.rows.map((r) => r.coef))]
  const [ci, setCi] = useState(Math.min(3, coefs.length - 1))
  const [kind, setKind] = useState('steering vector')
  const metric = name === 'weddings' ? 'wedding_rate' : 'positive_rate'
  const row = v.rows.find((r) => r.coef === coefs[ci] && (coefs[ci] === 0 || r.kind === kind)) ?? v.rows[0]
  const series = (k: string, color: string, dash?: string) => {
    const rs = v.rows.filter((r) => r.kind === k || r.coef === 0).sort((a, b) => a.coef - b.coef)
    return { name: k, x: rs.map((r) => r.coef), y: rs.map((r) => r[metric as 'wedding_rate']), color, dash }
  }
  return (
    <div>
      <HowTo>
        A steering vector is the difference between GPT-2-XL's activations on two short prompts (“{v.plus}” minus “{v.minus}”) at
        layer {v.layer}. Adding it (× c) while the model writes pushes the text toward the first prompt's meaning. The control adds
        a random vector of the same size.
      </HowTo>
      <div className="viz-controls">
        {names.map((n) => (
          <label key={n}>
            <input type="radio" checked={name === n} onChange={() => setName(n)} /> {n}
          </label>
        ))}
        <label>
          coefficient c = {coefs[ci]}
          <input type="range" min={0} max={coefs.length - 1} value={ci} onChange={(e) => setCi(+e.target.value)} />
        </label>
        <label>
          <input type="checkbox" checked={kind !== 'steering vector'} onChange={(e) => setKind(e.target.checked ? 'random vector (control)' : 'steering vector')} /> show random-vector control
        </label>
      </div>
      <div className="two-col">
        <LineChart
          series={[series('steering vector', 'var(--c-steering)'), series('random vector (control)', '#94a3b8', '5 4')]}
          xLabel="coefficient c"
          yLabel={name === 'weddings' ? 'completions mentioning weddings' : 'completions judged positive'}
          yDomain={[0, 1]}
          marker={coefs[ci]}
          fmtX={(x) => String(x)}
          height={260}
        />
        <div>
          {row.examples.map((e, i) => (
            <p key={i} style={{ fontSize: 14, margin: '6px 0' }}>
              <span style={{ opacity: 0.6 }}>{e.prompt}</span>
              <b>{e.completion}</b>
            </p>
          ))}
        </div>
      </div>
    </div>
  )
}
