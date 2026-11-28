// J-lens: layer × position slice grid (top token per cell) for the Jacobian lens vs the logit lens; rank of the
// unspoken intermediate token across layers; coordinate-swap results; the confirmatory rank test on 24 fresh prompts
// (added after the methods review, criterion pre-stated in reports/planned-runs.md).
import { useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import LineChart from '../primitives/LineChart'
import { HowTo, Tabs } from '../primitives/common'

type Cell = [string, number][]
interface Grid {
  text: string
  tokens: string[]
  layers: number[]
  grid: { jlens: Cell[][]; logit: Cell[][] }
  intermediates: string[]
  answer: string
}
interface D {
  grids: Record<string, Grid>
  ranks: Record<string, { jlens: Record<string, number[]>; logit: Record<string, number[]> }>
  swap: { layer: number; top5_success: number; random_top5: number; baseline_correct: number; n: number }[]
  confirm?: { prompt: string; intermediate: string; rank_j: number; rank_logit: number }[]
}

export default function JLens({ data }: VizProps) {
  const d = data as unknown as D
  const keys = Object.keys(d.grids)
  const [key, setKey] = useState(keys[0])
  const [lens, setLens] = useState<'jlens' | 'logit'>('jlens')
  const [tab, setTab] = useState<'grid' | 'ranks' | 'swap' | 'confirm'>('grid')
  const g = d.grids[key]
  const pin = new Set([...g.intermediates, g.answer].map((s) => s.trim()))
  const rows = [...g.layers.keys()].reverse()
  return (
    <div>
      <Tabs
        tabs={[
          ['grid', '1 · What each layer would say'],
          ['ranks', '2 · Unspoken intermediate steps'],
          ['swap', '3 · Swap a concept'],
          ...(d.confirm ? ([['confirm', '4 · Confirmatory test (24 prompts)']] as [typeof tab, string][]) : []),
        ]}
        value={tab}
        onChange={setTab}
      />
      <div className="viz-controls" style={{ display: tab === 'confirm' ? 'none' : undefined }}>
        <select className="sel" value={key} onChange={(e) => setKey(e.target.value)}>
          {keys.map((k) => (
            <option key={k} value={k}>
              {d.grids[k].text}
            </option>
          ))}
        </select>
      </div>
      {tab === 'grid' && (
        <>
          <HowTo>
            Each cell is one layer (rows, deepest on top) at one token (columns): the token the lens says that activation would
            produce. The J-lens maps each layer through an averaged Jacobian before reading it out; the logit lens reads raw
            activations. Highlighted: the unspoken intermediate ({g.intermediates.join(', ')}) and the answer ({g.answer}).
          </HowTo>
          <div className="viz-controls">
            <label>
              <input type="radio" checked={lens === 'jlens'} onChange={() => setLens('jlens')} /> J-lens
            </label>
            <label>
              <input type="radio" checked={lens === 'logit'} onChange={() => setLens('logit')} /> logit lens
            </label>
          </div>
          <div className="table-wrap">
            <table className="data" style={{ fontSize: 11, fontFamily: 'ui-monospace, monospace' }}>
              <thead>
                <tr>
                  <th>layer</th>
                  {g.tokens.map((t, i) => (
                    <th key={i}>{t}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((ri) => (
                  <tr key={ri}>
                    <td>{g.layers[ri]}</td>
                    {g.grid[lens][ri].map((cell, ti) => {
                      const [tok, p] = cell[0]
                      const hit = pin.has(tok.trim())
                      return (
                        <td key={ti} title={cell.map(([t, q]) => `${JSON.stringify(t)} ${(q * 100).toFixed(1)}%`).join('\n')} style={{ background: hit ? 'color-mix(in srgb, var(--c-lens) 55%, transparent)' : `rgba(124,58,237,${Math.min(0.6, p)})`, fontWeight: hit ? 700 : 400 }}>
                          {tok.replace(/\n/g, '⏎').slice(0, 8)}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {tab === 'ranks' && (
        <>
          <HowTo>Rank of each intermediate / answer token at the last position, layer by layer (1 = top prediction; log scale). Lower = the lens “sees” it.</HowTo>
          <LineChart
            series={Object.entries(d.ranks[key].jlens).flatMap(([w, r], i) => [
              { name: `J-lens: ${w}`, x: g.layers, y: r, color: ['#7c3aed', '#0891b2', '#16a34a'][i % 3], width: 2.5 },
              { name: `logit lens: ${w}`, x: g.layers, y: d.ranks[key].logit[w], color: ['#7c3aed', '#0891b2', '#16a34a'][i % 3], dash: '4 3', width: 1.5 },
            ])}
            yLog
            xLabel="layer"
            yLabel="rank at last position (log)"
            fmtX={(v) => String(v)}
          />
        </>
      )}
      {tab === 'swap' && (
        <>
          <HowTo>
            At the country token (e.g. “France” in “The capital of France is”) exchange the activation's coordinates along the J-lens
            directions of two countries, then check whether the target country's answer (e.g. Beijing) enters the top 5.
            Control: exchange with a random direction.
          </HowTo>
          <Bars
            bars={d.swap.flatMap((s) => [
              { label: `layer ${s.layer} · J-lens swap`, value: s.top5_success, color: 'var(--c-lens)' },
              { label: `layer ${s.layer} · random swap`, value: s.random_top5, color: '#94a3b8' },
            ])}
            xLabel="target answer in top-5"
            domain={[0, 1]}
          />
        </>
      )}
      {tab === 'confirm' && d.confirm && (
        <>
          <HowTo>
            24 new two-hop prompts, written before the lens was run on them. For each, the best rank (over layers) of the unspoken
            intermediate at the last position under each lens; the better lens is in bold. Pre-stated test: the J-lens wins more
            prompts and a sign test over non-tied prompts gives p &lt; 0.05.
          </HowTo>
          <div className="table-wrap">
            <table className="data" style={{ fontSize: 12 }}>
              <thead>
                <tr>
                  <th>Prompt</th>
                  <th>Intermediate</th>
                  <th>J-lens rank</th>
                  <th>logit-lens rank</th>
                </tr>
              </thead>
              <tbody>
                {d.confirm.map((c) => (
                  <tr key={c.prompt}>
                    <td>{c.prompt}</td>
                    <td>{c.intermediate}</td>
                    <td style={{ fontWeight: c.rank_j < c.rank_logit ? 700 : 400 }}>{c.rank_j}</td>
                    <td style={{ fontWeight: c.rank_logit < c.rank_j ? 700 : 400 }}>{c.rank_logit}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
