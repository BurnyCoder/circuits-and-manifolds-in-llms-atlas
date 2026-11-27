// Copy suppression entry: worked prompts (L10H7 attention to the earlier copy, direct effect, ablation) and the
// weight-based OV/QK evidence with a comparison against every other head in layers 10–11.
import { useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import TokenAttention from '../primitives/TokenAttention'
import { HowTo, Kpis, Tabs } from '../primitives/common'

interface Ex {
  text: string
  target: string
  tokens: string[]
  attn: number[]
  copy_pos: number[]
  attn_to_copy: number
  direct_effect: number
  p_clean: number
  p_ablated: number
}
interface D {
  rank_hist: number[]
  rank_edges: number[]
  controls: Record<string, { frac_top10: number; frac_bottom5pct: number; qk_diag_max: number }>
  examples: Ex[]
  summary: Record<string, number>
}

export default function CopySuppression({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'examples' | 'weights'>('examples')
  const [i, setI] = useState(0)
  const ex = d.examples[i]
  const s = d.summary
  const heads = [
    { label: '10.7 (copy suppression)', value: s.ov_top10, color: 'var(--c-attribution)' },
    ...Object.entries(d.controls)
      .sort((a, b) => b[1].frac_top10 - a[1].frac_top10)
      .slice(0, 8)
      .map(([k, v]) => ({ label: k, value: v.frac_top10, color: '#64748b' })),
  ]
  const maxH = Math.max(...d.rank_hist)
  return (
    <div>
      <Tabs
        tabs={[
          ['examples', '1 · Watch it suppress'],
          ['weights', '2 · Read it from the weights'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'examples' && (
        <>
          <HowTo>
            Pick a prompt. Earlier layers want to repeat a word already in the context (e.g. “love” in “All's fair in love
            and”). Head 10.7 attends back to that word (purple) and pushes its logit <b>down</b>. Removing the head
            (zero-ablation) makes the repeated word more likely.
          </HowTo>
          <select className="sel" value={i} onChange={(e) => setI(+e.target.value)}>
            {d.examples.map((e, k) => (
              <option key={k} value={k}>
                {e.text} → “{e.target.trim()}”
              </option>
            ))}
          </select>
          <TokenAttention
            tokens={ex.tokens}
            pattern={ex.tokens.map((_, k) => (k === ex.tokens.length - 1 ? ex.attn : ex.tokens.map(() => 0)))}
            initial={ex.tokens.length - 1}
            highlight={Object.fromEntries(ex.copy_pos.map((p) => [p, '#dc2626']))}
          />
          <Kpis
            items={[
              ['attention from last token to the earlier copy', ex.attn_to_copy.toFixed(2)],
              [`direct effect of 10.7 on logit(“${ex.target.trim()}”)`, ex.direct_effect.toFixed(2)],
              [`p(“${ex.target.trim()}”) with 10.7`, ex.p_clean.toFixed(4)],
              [`p(“${ex.target.trim()}”) with 10.7 ablated`, ex.p_ablated.toFixed(4)],
            ]}
          />
        </>
      )}
      {tab === 'weights' && (
        <>
          <HowTo>
            OV circuit: for each of {s.n_tokens} tokens, send its (effective) embedding through head 10.7's value/output
            matrices and the unembedding, then rank the effect on that same token among all 50,257 output tokens (rank 1 =
            pushed down the most). The histogram is over log₁₀(rank): mass at the far left means “attending to a token
            suppresses that token”.
          </HowTo>
          <svg viewBox="0 0 560 220" style={{ width: '100%', maxWidth: 560 }}>
            {d.rank_hist.map((h, k) => {
              const x = 40 + (k / d.rank_hist.length) * 500
              const bh = (h / maxH) * 170
              return <rect key={k} x={x} y={190 - bh} width={500 / d.rank_hist.length - 1} height={bh} fill={d.rank_edges[k + 1] <= 1.0001 ? 'var(--c-attribution)' : 'var(--c-circuit)'} opacity={0.85} />
            })}
            {[0, 1, 2, 3, 4].map((t) => (
              <text key={t} x={40 + (t / Math.log10(50257)) * 500} y={206} fontSize={11} textAnchor="middle" fill="currentColor">
                {10 ** t}
              </text>
            ))}
            <text x={290} y={219} fontSize={12} textAnchor="middle" fill="currentColor" opacity={0.75}>
              rank of own-token logit effect (log scale, 1 = most suppressed)
            </text>
          </svg>
          <Kpis
            items={[
              ['tokens whose own logit is in the 10 most suppressed', `${(s.ov_top10 * 100).toFixed(1)}%`],
              ['… in the bottom 5% of all tokens', `${(s.ov_bottom5 * 100).toFixed(1)}%`],
              ['QK: predicted token attends to its own copy (diag max)', `${(s.qk_diag * 100).toFixed(1)}%`],
              ['median of other heads in layers 10–11 (top-10 metric)', `${(s.ctrl_top10_median * 100).toFixed(1)}%`],
            ]}
          />
          <h3>Same OV metric for other late heads (control)</h3>
          <Bars bars={heads} xLabel="fraction of tokens whose own logit is in the 10 most suppressed" domain={[0, 1]} />
        </>
      )}
    </div>
  )
}
