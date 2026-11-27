// Induction heads entry: (1) watch a previous-token head and an induction head on repeated random tokens,
// (2) head-score maps for attn-only-2l and GPT-2 small vs a random-init control, (3) causal ablations,
// (4) phase change across Pythia-160M training checkpoints.
import { useMemo, useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import Heatmap, { Legend } from '../primitives/Heatmap'
import LineChart from '../primitives/LineChart'
import TokenAttention from '../primitives/TokenAttention'
import { HowTo, Kpis, Tabs, seqScale } from '../primitives/common'

type Scores = { prev: number[][]; induction: number[][]; duplicate: number[][] }
interface D {
  a2: {
    scores: Scores
    rand_scores: Scores
    loss_curve: number[]
    ablations: Record<string, [number, number, number]>
    example: { tokens: string[]; seq_len: number; patterns: Record<string, number[][]> }
  }
  gpt2: { scores: Scores }
  pythia: { step: number; tokens_B: number; max_induction: number; icl: number; rep_loss_first: number; rep_loss_second: number }[]
}

const METRICS: [keyof Scores, string][] = [
  ['induction', 'Induction score'],
  ['prev', 'Previous-token score'],
  ['duplicate', 'Duplicate-token score'],
]

export default function Induction({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'watch' | 'scores' | 'ablate' | 'phase'>('watch')
  const heads = Object.keys(d.a2.example.patterns)
  const [head, setHead] = useState(heads[heads.length - 1])
  const [metric, setMetric] = useState<keyof Scores>('induction')
  const L = d.a2.example.seq_len
  const color = useMemo(() => seqScale(1, undefined), [])

  return (
    <div>
      <Tabs
        tabs={[
          ['watch', '1 · Watch it copy'],
          ['scores', '2 · Find the heads'],
          ['ablate', '3 · Knock them out'],
          ['phase', '4 · Phase change in training'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'watch' && (
        <>
          <HowTo>
            The sequence is {L} random tokens, then the same {L} tokens again. Hover any token in the <b>second half</b>: the
            induction head looks back at the token that came <i>right after</i> the earlier copy of the current token, which
            is exactly the token that comes next. The previous-token head (layer 0) always looks one step back; that is
            what lets the layer-1 head find “the token after”.
          </HowTo>
          <div className="viz-controls">
            {heads.map((h) => (
              <label key={h}>
                <input type="radio" checked={head === h} onChange={() => setHead(h)} /> head {h} {h.startsWith('0.') ? '(previous-token)' : '(induction)'}
              </label>
            ))}
          </div>
          <TokenAttention tokens={d.a2.example.tokens} pattern={d.a2.example.patterns[head]} initial={L + 6} />
          <h3>Loss per position (attn-only-2l, mean over sequences)</h3>
          <LineChart
            series={[{ name: 'next-token loss (nats)', x: d.a2.loss_curve.map((_, i) => i + 1), y: d.a2.loss_curve, color: 'var(--c-circuit)' }]}
            xLabel="position in sequence"
            yLabel="loss (nats)"
            bands={[{ x0: L + 1, x1: 2 * L, label: 'repeated half', color: '#22c55e' }]}
            fmtX={(v) => String(v)}
            height={240}
          />
          <p className="viz-note">Random tokens are unpredictable (≈ log vocab nats) until they repeat; then the induction circuit copies them.</p>
        </>
      )}
      {tab === 'scores' && (
        <>
          <HowTo>
            Each cell is one attention head (row = layer, column = head). The score is its average attention along the
            relevant diagonal on repeated random tokens (1 = always attends to the predicted source). Compare a trained
            model with the same architecture at random initialization.
          </HowTo>
          <div className="viz-controls">
            {METRICS.map(([k, l]) => (
              <label key={k}>
                <input type="radio" checked={metric === k} onChange={() => setMetric(k)} /> {l}
              </label>
            ))}
          </div>
          <div className="two-col">
            <div>
              <h3>attn-only-2l (trained)</h3>
              <Heatmap values={d.a2.scores[metric]} color={color} cell={34} yLabels={['L0', 'L1']} xLabels={[...Array(8).keys()].map(String)} xTitle="head" tooltip={(r, c, v) => `head ${r}.${c}: ${v.toFixed(3)}`} />
              <h3>attn-only-2l (random init, control)</h3>
              <Heatmap values={d.a2.rand_scores[metric]} color={color} cell={34} yLabels={['L0', 'L1']} xLabels={[...Array(8).keys()].map(String)} xTitle="head" tooltip={(r, c, v) => `head ${r}.${c}: ${v.toFixed(3)}`} />
            </div>
            <div>
              <h3>GPT-2 small</h3>
              <Heatmap values={d.gpt2.scores[metric]} color={color} cell={24} yLabels={[...Array(12).keys()].map((i) => `L${i}`)} xLabels={[...Array(12).keys()].map(String)} xTitle="head" yTitle="layer" tooltip={(r, c, v) => `head ${r}.${c}: ${v.toFixed(3)}`} />
            </div>
          </div>
          <Legend color={color} lo={0} hi={1} label={METRICS.find(([k]) => k === metric)![1]} />
        </>
      )}
      {tab === 'ablate' && (
        <>
          <HowTo>
            Causal test: zero out a head's output and measure the loss on the repeated half (lower = better copying).
            Error bars are 95% bootstrap confidence intervals over sequences. The control ablates the same number of
            random non-induction heads in layer 1.
          </HowTo>
          <Bars
            bars={Object.entries(d.a2.ablations).map(([k, [v, lo, hi]], i) => ({
              label: k,
              value: v,
              lo,
              hi,
              color: ['#16a34a', '#dc2626', '#d97706', '#64748b'][i % 4],
            }))}
            xLabel="loss on repeated tokens (nats)"
          />
          <Kpis
            items={[
              ['clean repeat loss', d.a2.ablations['clean'][0].toFixed(2)],
              ['induction heads ablated', d.a2.ablations['ablate induction heads'][0].toFixed(2)],
              ['prev-token head ablated', d.a2.ablations['ablate prev-token head'][0].toFixed(2)],
            ]}
          />
        </>
      )}
      {tab === 'phase' && (
        <>
          <HowTo>
            Pythia-160M was trained on 300B tokens with checkpoints saved along the way. For each checkpoint we measure the
            strongest induction head and the in-context-learning (ICL) score: loss on the 500th token minus loss on the 50th
            (more negative = the model uses more context). Both change abruptly early in training (x axis is log-scaled).
          </HowTo>
          <LineChart
            series={[{ name: 'max induction score (any head)', x: d.pythia.map((r) => Math.max(r.tokens_B, 0.05)), y: d.pythia.map((r) => r.max_induction), color: 'var(--c-circuit)' }]}
            xLog
            xLabel="training tokens (billions, log scale; step 0 drawn at 0.05B)"
            yLabel="induction score"
            fmtX={(v) => `${v}B`}
            height={260}
          />
          <LineChart
            series={[{ name: 'ICL score = loss(500th) − loss(50th)', x: d.pythia.map((r) => Math.max(r.tokens_B, 0.05)), y: d.pythia.map((r) => r.icl), color: 'var(--c-manifold)' }]}
            xLog
            xLabel="training tokens (billions, log scale)"
            yLabel="ICL score (nats)"
            fmtX={(v) => `${v}B`}
            height={260}
          />
        </>
      )}
    </div>
  )
}
