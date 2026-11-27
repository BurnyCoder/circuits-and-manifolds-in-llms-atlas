// Attribution graph viewer: public Gemma-2-2B circuit-tracer graphs (Neuronpedia), aggregated to labelled supernodes.
// x = token position, y = layer (embeddings at the bottom, output logits on top); edge width ∝ |direct effect|
// (normalized per target), blue = excitatory, red = inhibitory. Hover a node to highlight its connections.
import { useMemo, useState } from 'react'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import { HowTo, Kpis, Tabs } from '../primitives/common'

interface GNode {
  id: string
  label: string
  kind: 'embed' | 'feature' | 'super' | 'logit' | 'error'
  layer: number
  ctx: number
  members: string[]
  prob?: number
}
interface Graph {
  slug: string
  title: string
  prompt: string
  tokens: string[]
  nodes: GNode[]
  edges: { s: string; t: string; w: number }[]
  n_nodes: number
  n_links: number
  error_share_into_logits: number
  np_url: string
}
interface D {
  graphs: Graph[]
  two_hop: { slug: string; path: number; direct: number }[]
  overlap: Record<string, number>
}

const KIND_COLOR: Record<string, string> = { super: '#7c3aed', feature: '#a78bfa', logit: '#16a34a', embed: '#64748b', error: '#94a3b8' }

function layout(g: Graph, W: number, H: number) {
  const nTok = g.tokens.length
  const colW = (W - 220) / nTok // leave room on the right for supernode labels at the last token
  const yOf = (layer: number) => (layer < 0 ? H - 34 : layer > 50 ? 26 : H - 60 - (layer / 26) * (H - 110))
  const pos: Record<string, [number, number]> = {}
  const byCol: Record<number, GNode[]> = {}
  // Output logits get their own evenly spaced row across the top (they all sit at the last token position).
  const logits = g.nodes.filter((n) => n.kind === 'logit').sort((a, b) => (b.prob ?? 0) - (a.prob ?? 0))
  logits.forEach((n, i) => (pos[n.id] = [W / 2 + (i - (logits.length - 1) / 2) * Math.min(170, (W - 120) / logits.length), 26]))
  for (const n of g.nodes) if (n.kind !== 'logit') (byCol[Math.round(n.ctx)] ??= []).push(n)
  for (const [c, ns] of Object.entries(byCol)) {
    ns.sort((a, b) => a.layer - b.layer)
    const placed: [number, number][] = []
    for (const n of ns) {
      let x = 60 + Number(c) * colW + colW / 2
      const y = yOf(n.layer)
      let k = 0
      while (placed.some(([px, py]) => Math.abs(px - x) < 18 && Math.abs(py - y) < 16) && k < 8) {
        k++
        x = 60 + Number(c) * colW + colW / 2 + (k % 2 ? 1 : -1) * Math.ceil(k / 2) * 20
      }
      placed.push([x, y])
      pos[n.id] = [x, y]
    }
  }
  return { pos, colW, yOf }
}

export default function AttributionGraph({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'graph' | 'checks'>('graph')
  const [gi, setGi] = useState(0)
  const [hover, setHover] = useState<string | null>(null)
  const [minW, setMinW] = useState(0.15)
  const g = d.graphs[gi]
  const W = 900
  const H = 520
  const { pos, colW, yOf } = useMemo(() => layout(g, W, H), [g])
  const byId = useMemo(() => Object.fromEntries(g.nodes.map((n) => [n.id, n])), [g])
  const maxIn = useMemo(() => {
    const m: Record<string, number> = {}
    for (const e of g.edges) m[e.t] = Math.max(m[e.t] ?? 0, Math.abs(e.w))
    return m
  }, [g])
  const edges = g.edges.filter((e) => Math.abs(e.w) / (maxIn[e.t] || 1) >= minW && pos[e.s] && pos[e.t])
  const connected = hover ? new Set(edges.filter((e) => e.s === hover || e.t === hover).flatMap((e) => [e.s, e.t])) : null
  const hn = hover ? byId[hover] : null
  return (
    <div>
      <Tabs
        tabs={[
          ['graph', '1 · Explore a graph'],
          ['checks', '2 · Structural checks'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'graph' && (
        <>
          <HowTo>
            Each graph explains one next-token prediction of Gemma-2-2B. Bottom row: the prompt's tokens. Purple nodes: groups
            of transcoder features (“supernodes”, human-labelled where curated) at their layer (height) and token (column).
            Green: output tokens. Lines: direct causal effects (blue = boosts, red = suppresses). Hover a node to follow its
            inputs and outputs; open the original on Neuronpedia for the full, feature-level graph.
          </HowTo>
          <div className="viz-controls">
            <select className="sel" value={gi} onChange={(e) => setGi(+e.target.value)}>
              {d.graphs.map((x, i) => (
                <option key={x.slug} value={i}>
                  {x.title}
                </option>
              ))}
            </select>
            <label>
              hide edges weaker than {(minW * 100).toFixed(0)}% of the strongest input
              <input type="range" min={0.05} max={0.6} step={0.05} value={minW} onChange={(e) => setMinW(+e.target.value)} />
            </label>
            <a href={g.np_url} target="_blank" rel="noopener">
              open on Neuronpedia ↗
            </a>
          </div>
          <p className="viz-note">
            Prompt: <code>{g.prompt.replace('<bos>', '')}</code>
          </p>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img" aria-label="attribution graph">
            {[0, 5, 10, 15, 20, 25].map((l) => (
              <g key={l}>
                <line x1={50} x2={W - 10} y1={yOf(l)} y2={yOf(l)} stroke="currentColor" opacity={0.07} />
                <text x={44} y={yOf(l) + 4} fontSize={10} textAnchor="end" fill="currentColor" opacity={0.6}>
                  L{l}
                </text>
              </g>
            ))}
            {edges.map((e, i) => {
              const [x1, y1] = pos[e.s]
              const [x2, y2] = pos[e.t]
              const rel = Math.abs(e.w) / (maxIn[e.t] || 1)
              const on = !connected || (connected.has(e.s) && connected.has(e.t) && (e.s === hover || e.t === hover))
              return (
                <path
                  key={i}
                  d={`M${x1},${y1} C${x1},${(y1 + y2) / 2} ${x2},${(y1 + y2) / 2} ${x2},${y2}`}
                  fill="none"
                  stroke={e.w > 0 ? '#2563eb' : '#dc2626'}
                  strokeWidth={0.6 + 4 * rel}
                  opacity={on ? 0.25 + 0.6 * rel : 0.04}
                />
              )
            })}
            {g.tokens.map((t, i) => (
              <text key={i} x={60 + i * colW + colW / 2} y={H - 8} fontSize={11} textAnchor="middle" fill="currentColor">
                {t.replace('<bos>', '⟨bos⟩').slice(0, 10)}
              </text>
            ))}
            {g.nodes.map((n) => {
              const p = pos[n.id]
              if (!p) return null
              const dim = connected && !connected.has(n.id) && n.id !== hover
              const r = n.kind === 'super' ? 9 : n.kind === 'logit' ? 8 : n.kind === 'embed' ? 6 : 6
              return (
                <g key={n.id} onMouseEnter={() => setHover(n.id)} onMouseLeave={() => setHover(null)} style={{ cursor: 'pointer' }} opacity={dim ? 0.25 : 1}>
                  {n.kind === 'logit' || n.kind === 'super' ? (
                    <rect x={p[0] - r} y={p[1] - r} width={2 * r} height={2 * r} rx={4} fill={KIND_COLOR[n.kind]} stroke="var(--panel)" />
                  ) : (
                    <circle cx={p[0]} cy={p[1]} r={r} fill={KIND_COLOR[n.kind]} stroke="var(--panel)" />
                  )}
                  {n.kind === 'logit' && (
                    <text x={p[0]} y={p[1] - r - 4} fontSize={11} textAnchor="middle" fill="currentColor" fontWeight={600}>
                      {n.label.replace('→ Output ', '').replace(/\s*\(p=.*\)/, '')} {n.prob ? `${(n.prob * 100).toFixed(0)}%` : ''}
                    </text>
                  )}
                  {n.kind === 'super' && (
                    <text x={p[0] + r + 3} y={p[1] + 4} fontSize={10.5} fill="currentColor">
                      {n.label.slice(0, 28)}
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
          <div className="viz-controls">
            {Object.entries({ super: 'labelled supernode', feature: 'transcoder feature', logit: 'output token', embed: 'input token', error: 'error node' }).map(([k, l]) => (
              <span key={k} className="tag" style={{ background: KIND_COLOR[k], color: '#fff' }}>
                {l}
              </span>
            ))}
          </div>
          {hn && (
            <p className="viz-note">
              <b>{hn.label || hn.id}</b> · {hn.kind} · layer {hn.layer < 0 ? 'embedding' : hn.layer > 50 ? 'output' : hn.layer.toFixed(1)} · token “{g.tokens[Math.round(hn.ctx)]}” ·{' '}
              {hn.members.length} feature{hn.members.length > 1 ? 's' : ''}
              {hn.prob ? ` · p = ${hn.prob.toFixed(3)}` : ''}
            </p>
          )}
          <Kpis
            items={[
              ['nodes / links in the full graph', `${g.n_nodes} / ${g.n_links.toLocaleString()}`],
              ['shown after grouping', g.nodes.length],
              ['share of logit inputs from error nodes', `${(g.error_share_into_logits * 100).toFixed(0)}%`],
            ]}
          />
        </>
      )}
      {tab === 'checks' && (
        <>
          <HowTo>
            Two of the paper's claims, checked on the public Gemma-2-2B graphs. Two-hop recall: strength of the best positive
            path “city token → intermediate concept (state / country) → answer” vs the direct city → answer edge. Multilingual:
            overlap of the transcoder features used by the English, French and Chinese antonym graphs, by layer range.
          </HowTo>
          <Bars
            bars={d.two_hop.flatMap((t) => [
              { label: `${t.slug.replace('gemma-fact-', '')} · two-hop`, value: t.path, color: '#7c3aed' },
              { label: `${t.slug.replace('gemma-fact-', '')} · direct`, value: t.direct, color: '#94a3b8' },
            ])}
            xLabel="path strength (weakest edge on the path, summed attribution units)"
          />
          <Bars bars={Object.entries(d.overlap).map(([k, v]) => ({ label: k, value: v, color: k.startsWith('middle') ? '#7c3aed' : '#94a3b8' }))} xLabel="feature overlap of the 3 languages (Jaccard)" domain={[0, Math.max(...Object.values(d.overlap), 0.2)]} />
        </>
      )}
    </div>
  )
}
