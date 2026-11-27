// Calendar circles: day/month representations in GPT-2 small (raw residual and SAE reconstruction), how SAE latents
// tile the circle, a Qwen3-1.7B layer sweep, and the causal "move the day along the circle" test.
import * as d3 from 'd3'
import { useMemo, useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import Heatmap, { Legend } from '../primitives/Heatmap'
import LineChart from '../primitives/LineChart'
import Plot3D from '../primitives/Plot3D'
import Scatter2D from '../primitives/Scatter2D'
import { HowTo, Kpis, Tabs, seqScale } from '../primitives/common'

interface Circle {
  points: number[][]
  labels: number[]
  means: number[][]
  var: number[]
  score: number
  p: number
}
interface D {
  days: string[]
  months: string[]
  gpt2: { days: Circle & { sae: Circle & { features: number[]; tiling: number[][] } }; months: Circle & { sae: Circle & { features: number[]; tiling: number[][] } }; control: { n: number; score: number; p: number }[] }
  qwen: {
    geo: Record<'days' | 'months', (Circle & { layer: number })[]>
    accuracy: number
    causal: { layer: number; success: number; success_random_plane: number; n: number }[]
    n_prompts: number
  }
}

const hue = (i: number, n: number) => d3.interpolateRainbow(i / n)

function CircleView({ c, names, three }: { c: Circle; names: string[]; three: boolean }) {
  const n = names.length
  const colors = c.labels.map((l) => hue(l, n))
  if (three)
    return <Plot3D traces={[{ x: c.points.map((p) => p[0]), y: c.points.map((p) => p[1]), z: c.points.map((p) => p[2]), color: colors, size: 4, text: c.labels.map((l) => names[l].trim()) }]} height={380} />
  return (
    <Scatter2D
      xy={c.points.map((p) => [p[0], p[1]] as [number, number])}
      colors={colors}
      size={3.2}
      alpha={0.55}
      labels={c.means.map((m, i) => ({ xy: [m[0], m[1]] as [number, number], text: names[i].trim().slice(0, 3), color: hue(i, n) }))}
      polygon={c.means.map((m) => [m[0], m[1]] as [number, number])}
    />
  )
}

export default function CalendarCircles({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'gpt2' | 'sae' | 'qwen' | 'causal'>('gpt2')
  const [kind, setKind] = useState<'days' | 'months'>('days')
  const [view, setView] = useState<'raw' | 'sae'>('raw')
  const [three, setThree] = useState(false)
  const names = kind === 'days' ? d.days : d.months
  const g = d.gpt2[kind]
  const c = view === 'raw' ? g : g.sae
  const qs = d.qwen.geo[kind]
  const [qi, setQi] = useState(Math.floor(qs.length / 2))
  const tileMax = Math.max(...g.sae.tiling.flat())
  const tileColor = useMemo(() => seqScale(tileMax, d3.interpolateMagma), [tileMax])
  const toggles = (
    <div className="viz-controls">
      <label>
        <input type="radio" checked={kind === 'days'} onChange={() => setKind('days')} /> days of the week
      </label>
      <label>
        <input type="radio" checked={kind === 'months'} onChange={() => setKind('months')} /> months
      </label>
      <label>
        <input type="checkbox" checked={three} onChange={(e) => setThree(e.target.checked)} /> 3-D
      </label>
    </div>
  )
  return (
    <div>
      <Tabs
        tabs={[
          ['gpt2', '1 · The circle in GPT-2'],
          ['sae', '2 · SAE latents tile it'],
          ['qwen', '3 · Layer by layer (Qwen3)'],
          ['causal', '4 · Move the day (causal)'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'gpt2' && (
        <>
          <HowTo>
            Each dot is GPT-2 small's layer-7 residual stream at a day (or month) token in one of 16 sentences, with the
            sentence's average removed so only “which day” is left; axes are the top principal components. Big dots are the
            averages per day, joined in calendar order: they go <b>around a circle</b>, Monday → Sunday → Monday.
          </HowTo>
          {toggles}
          <div className="viz-controls">
            <label>
              <input type="radio" checked={view === 'raw'} onChange={() => setView('raw')} /> raw activations
            </label>
            <label>
              <input type="radio" checked={view === 'sae'} onChange={() => setView('sae')} /> reconstructed from {g.sae.features.length} SAE latents
            </label>
          </div>
          <CircleView c={c} names={names} three={three} />
          <Kpis
            items={[
              ['cyclic order of the class means (1 = perfect)', c.score.toFixed(2)],
              ['p-value vs random arrangement', c.p < 0.001 ? '< 0.001' : c.p.toFixed(3)],
              ['variance in PC1–PC3', `${(c.var.slice(0, 3).reduce((a, b) => a + b, 0) * 100).toFixed(0)}%`],
              ['random-word sets that look cyclic (p < .05)', `${d.gpt2.control.filter((x) => x.p < 0.05).length}/${d.gpt2.control.length}`],
            ]}
          />
        </>
      )}
      {tab === 'sae' && (
        <>
          <HowTo>
            Rows = sparse-autoencoder latents (GPT-2 small, layer 7) that fire most on {kind}; columns = the {kind}. Each latent
            prefers a few <b>neighbouring</b> {kind} — together they tile the circle like overlapping arcs (cf. “capture vs.
            dilution” in Bhalla et al. 2026).
          </HowTo>
          {toggles}
          <Heatmap
            values={g.sae.tiling}
            color={tileColor}
            cell={kind === 'days' ? 30 : 22}
            xLabels={names.map((s) => s.trim().slice(0, 3))}
            yLabels={g.sae.features.map(String)}
            xTitle={kind}
            yTitle="SAE latent"
            maxLabelEvery={1}
            tooltip={(r, col, v) => `latent ${g.sae.features[r]} on ${names[col].trim()}: ${v.toFixed(2)}`}
          />
          <Legend color={tileColor} lo={0} hi={tileMax} label="mean latent activation" />
        </>
      )}
      {tab === 'qwen' && (
        <>
          <HowTo>The same analysis in Qwen3-1.7B at different depths. Drag the slider: where does the calendar circle appear?</HowTo>
          {toggles}
          <div className="viz-controls">
            <label>
              layer {qs[qi].layer}
              <input type="range" min={0} max={qs.length - 1} value={qi} onChange={(e) => setQi(+e.target.value)} />
            </label>
          </div>
          <div className="two-col">
            <CircleView c={{ ...qs[qi], labels: d.qwen.geo[kind][qi].points.map((_, i) => i % names.length) }} names={names} three={three} />
            <LineChart
              series={[{ name: 'cyclic order score', x: qs.map((q) => q.layer), y: qs.map((q) => q.score), color: 'var(--c-manifold)' }]}
              xLabel="layer"
              yLabel="cyclic order (1 = perfect)"
              yDomain={[0, 1]}
              fmtX={(v) => String(v)}
              marker={qs[qi].layer}
              height={260}
            />
          </div>
        </>
      )}
      {tab === 'causal' && (
        <>
          <HowTo>
            Prompt: “Let's do some day of the week math. Two days from Monday is …”. At the Monday token we move the activation
            <b> along the circle</b> to another day's position (e.g. Wednesday) and check whether the model's answer shifts
            accordingly (Friday instead of Wednesday). Control: the same-size move in a random 2-D plane.
          </HowTo>
          <Bars
            bars={d.qwen.causal.flatMap((r) => [
              { label: `layer ${r.layer} · circle plane`, value: r.success, color: 'var(--c-manifold)' },
              { label: `layer ${r.layer} · random plane`, value: r.success_random_plane, color: '#94a3b8' },
            ])}
            xLabel="fraction of prompts whose answer moves to the target day"
            domain={[0, 1]}
          />
          <Kpis
            items={[
              ['clean day-arithmetic accuracy', `${(d.qwen.accuracy * 100).toFixed(0)}% of ${d.qwen.n_prompts}`],
              ['chance (7 days)', '14%'],
            ]}
          />
        </>
      )}
    </div>
  )
}
