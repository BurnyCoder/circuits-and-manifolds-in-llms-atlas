// SAE features: pick a concept → its GPT-2 SAE feature: activation on the probe sentence, top WikiText contexts,
// logit effects, and steering completions by strength (with a random-direction control).
import { useState } from 'react'
import type { VizProps } from '../registry'
import LineChart from '../primitives/LineChart'
import { HowTo, Tabs } from '../primitives/common'

interface C {
  feature: number
  probe_tokens: string[]
  probe_acts: number[]
  contexts: { tokens: string[]; acts: number[]; max: number }[]
  logit_up: string[]
  logit_down: string[]
  steer: { alpha: number; kind: string; on_topic: number; examples: string[] }[]
  neuronpedia: string
}
interface D {
  concepts: Record<string, C>
  layer: number
}

function Tokens({ tokens, acts }: { tokens: string[]; acts: number[] }) {
  const mx = Math.max(...acts, 1e-6)
  return (
    <div className="token-strip">
      {tokens.map((t, i) => (
        <span key={i} title={acts[i].toFixed(2)} style={{ background: `rgba(219,39,119,${Math.min(1, acts[i] / mx) * 0.8})` }}>
          {t.replace(/\n/g, '⏎') || '·'}
        </span>
      ))}
    </div>
  )
}

export default function SaeFeatures({ data }: VizProps) {
  const d = data as unknown as D
  const names = Object.keys(d.concepts)
  const [name, setName] = useState(names[0])
  const [tab, setTab] = useState<'feature' | 'steer'>('feature')
  const c = d.concepts[name]
  const alphas = [...new Set(c.steer.map((s) => s.alpha))]
  const [ai, setAi] = useState(Math.min(3, alphas.length - 1))
  const row = c.steer.find((s) => s.alpha === alphas[ai] && s.kind === 'feature') ?? c.steer[0]
  return (
    <div>
      <div className="viz-controls">
        {names.map((n) => (
          <button key={n} className={`chip ${n === name ? 'on' : ''}`} onClick={() => setName(n)}>
            {n}
          </button>
        ))}
        <a href={c.neuronpedia} target="_blank" rel="noopener">
          feature #{c.feature} on Neuronpedia ↗
        </a>
      </div>
      <Tabs
        tabs={[
          ['feature', '1 · What the feature detects'],
          ['steer', '2 · Steer with it'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'feature' && (
        <>
          <HowTo>Pink = activation of SAE latent #{c.feature} (GPT-2 small, layer {d.layer}) on each token. First the probe sentence, then the strongest activations found in WikiText.</HowTo>
          <Tokens tokens={c.probe_tokens} acts={c.probe_acts} />
          <h3>Top-activating contexts</h3>
          {c.contexts.map((x, i) => (
            <div key={i} style={{ margin: '6px 0' }}>
              <Tokens tokens={x.tokens} acts={x.acts} />
            </div>
          ))}
          <p className="viz-note">
            <b>Promotes:</b> {c.logit_up.map((t) => JSON.stringify(t)).join(' ')} · <b>Suppresses:</b> {c.logit_down.map((t) => JSON.stringify(t)).join(' ')}
          </p>
        </>
      )}
      {tab === 'steer' && (
        <>
          <HowTo>Add α × the feature's decoder direction to every position at layer {d.layer} while GPT-2 writes from neutral prompts (“Golden Gate GPT-2”). Control: a random direction of the same size.</HowTo>
          <div className="viz-controls">
            <label>
              α = {alphas[ai]}
              <input type="range" min={0} max={alphas.length - 1} value={ai} onChange={(e) => setAi(+e.target.value)} />
            </label>
          </div>
          <div className="two-col">
            <LineChart
              series={['feature', 'random'].map((k, i) => {
                const rs = c.steer.filter((s) => s.kind === k || s.alpha === 0).sort((a, b) => a.alpha - b.alpha)
                return { name: k === 'feature' ? 'feature direction' : 'random direction', x: rs.map((s) => s.alpha), y: rs.map((s) => s.on_topic), color: i ? '#94a3b8' : 'var(--c-feature)', dash: i ? '5 4' : undefined }
              })}
              xLabel="steering strength α"
              yLabel="completions on the concept"
              yDomain={[0, 1]}
              marker={alphas[ai]}
              fmtX={(v) => String(v)}
              height={240}
            />
            <div>
              {row.examples.map((e, i) => (
                <p key={i} style={{ fontSize: 14 }}>
                  …{e}
                </p>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
