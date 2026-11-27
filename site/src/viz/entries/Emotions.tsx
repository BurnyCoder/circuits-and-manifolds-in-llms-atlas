// Emotion concepts: 2-D map of emotion vectors (PC1 × PC2) colored by k-means cluster or by human valence, the
// Tylenol dose sweep, the preference-steering test, and example stories the model wrote.
import * as d3 from 'd3'
import { useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import LineChart from '../primitives/LineChart'
import Scatter2D from '../primitives/Scatter2D'
import { HowTo, Kpis, Tabs, cat10 } from '../primitives/common'

interface D {
  emotions: string[]
  pcs: number[][]
  var: number[]
  valence: number[]
  arousal: number[]
  clusters: number[]
  sweep: { dose: number[]; afraid: number[]; calm: number[] }
  pref: Record<string, number>
  examples: { emotion: string; story: string }[]
  summary: Record<string, number>
}

export default function Emotions({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'map' | 'sweep' | 'steer' | 'stories'>('map')
  const [color, setColor] = useState<'cluster' | 'valence' | 'arousal'>('valence')
  const vmin = Math.min(...d.valence)
  const vmax = Math.max(...d.valence)
  const amin = Math.min(...d.arousal)
  const amax = Math.max(...d.arousal)
  const colors = d.emotions.map((_, i) =>
    color === 'cluster' ? cat10[d.clusters[i] % 10] : color === 'valence' ? d3.interpolateRdYlGn((d.valence[i] - vmin) / (vmax - vmin)) : d3.interpolatePlasma((d.arousal[i] - amin) / (amax - amin)),
  )
  return (
    <div>
      <Tabs
        tabs={[
          ['map', '1 · The emotion map'],
          ['sweep', '2 · Danger raises fear'],
          ['steer', '3 · Feelings shift choices'],
          ['stories', '4 · Stories it wrote'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'map' && (
        <>
          <HowTo>
            Each dot is one emotion's vector in Qwen3-1.7B (average activation while reading stories about it, minus the average
            over all emotions). The two main axes of variation: color by human-rated valence (green = pleasant) or arousal
            (bright = activated), or by unsupervised cluster.
          </HowTo>
          <div className="viz-controls">
            {(['valence', 'arousal', 'cluster'] as const).map((c) => (
              <label key={c}>
                <input type="radio" checked={color === c} onChange={() => setColor(c)} /> color by {c}
              </label>
            ))}
          </div>
          <Scatter2D xy={d.pcs.map((p) => [p[0], p[1]] as [number, number])} colors={colors} size={5} labels={d.pcs.map((p, i) => ({ xy: [p[0], p[1]] as [number, number], text: d.emotions[i], color: colors[i] }))} width={560} height={480} square={false} />
          <Kpis
            items={[
              ['PC1 / PC2 variance', `${(d.var[0] * 100).toFixed(0)}% / ${(d.var[1] * 100).toFixed(0)}%`],
              ['|r| PC1 vs human valence', d.summary.r_valence.toFixed(2)],
              ['|r| PC2 vs human arousal', d.summary.r_arousal.toFixed(2)],
              ['shuffled-label control (95th pct)', d.summary.null_valence_p95.toFixed(2)],
            ]}
          />
        </>
      )}
      {tab === 'sweep' && (
        <>
          <HowTo>“I just took X mg of Tylenol… should I be worried?” — projection of the model's activation at the end of the user turn onto the “afraid” and “calm” vectors as the dose grows.</HowTo>
          <LineChart
            series={[
              { name: 'afraid', x: d.sweep.dose, y: d.sweep.afraid, color: '#dc2626', width: 3 },
              { name: 'calm', x: d.sweep.dose, y: d.sweep.calm, color: '#2563eb', width: 3 },
            ]}
            xLog
            xLabel="dose (mg, log scale)"
            yLabel="projection on emotion vector"
            fmtX={(v) => String(v)}
          />
        </>
      )}
      {tab === 'steer' && (
        <>
          <HowTo>“Would you rather (A) … or (B) …?” over 120 pairs of everyday activities. Adding the blissful vector to the words of option A should make A more attractive; hostile should make it less. Random vector = control.</HowTo>
          <Bars bars={Object.entries(d.pref).map(([k, v]) => ({ label: k, value: v, color: k === 'blissful' ? '#16a34a' : k === 'hostile' ? '#dc2626' : '#94a3b8' }))} xLabel="mean probability of choosing option A" domain={[0, 1]} />
        </>
      )}
      {tab === 'stories' && (
        <div>
          {d.examples.map((e, i) => (
            <p key={i} style={{ fontSize: 14 }}>
              <span className="tag cat" style={{ ['--cat-color' as string]: 'var(--c-steering)' }}>
                {e.emotion}
              </span>{' '}
              {e.story}
            </p>
          ))}
        </div>
      )}
    </div>
  )
}
