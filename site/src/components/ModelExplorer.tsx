// Model Explorer: clickable architecture diagrams of every model in the atlas. Each head, MLP, residual-stream
// layer, CNN stage or (for closed models) depth band that hosts a circuit/manifold is highlighted in its category
// color; click it to see which findings live there and jump to them. For GPT-2 small, heads can also be colored by
// scores measured in this repo, and a clicked head shows its real attention pattern.
// Global context: locations come from entry frontmatter (`locations`, see src/lib/types.ts) plus measured
// induction / previous-token heads from public/data/gpt2-map.json.
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { entries } from '../lib/content'
import { useData } from '../lib/data'
import { CATEGORY } from '../lib/labels'
import { MODELS, modelById, type ModelSpec } from '../lib/models'
import type { Entry, Location } from '../lib/types'
import Heatmap from '../viz/primitives/Heatmap'
import { seqScale } from '../viz/primitives/common'
import { CategoryTag, FidelityBadge } from './Badges'

interface Hit {
  loc: Location
  entry: Entry
}
interface MapData {
  scores: { prev: number[][]; induction: number[][]; duplicate: number[][] }
  patterns: Record<string, { tokens: string[]; pattern: number[][][][] }>
}
type Overlay = 'findings' | 'induction' | 'prev' | 'duplicate'

const DEPTH_ROWS = 20 // closed models: rows = 5% depth slices

/** Does a location cover the drawn component `key`? Keys: head:L:H, attn:L, mlp:L, resid:L, embed, unembed, stage:NAME, depth:I */
function covers(loc: Location, key: string): boolean {
  const [kind, a, b] = key.split(':')
  const L = Number(a)
  switch (loc.part) {
    case 'head':
      return (kind === 'head' && loc.layer === L && loc.head === Number(b)) || (kind === 'attn' && loc.layer === L)
    case 'attn':
      return (kind === 'attn' || kind === 'head') && loc.layer === L
    case 'mlp':
      return kind === 'mlp' && loc.layer === L
    case 'resid': {
      if (kind !== 'resid') return false
      const [lo, hi] = loc.layers ?? [loc.layer ?? -99, loc.layer ?? -99]
      return L >= lo && L <= hi
    }
    case 'embed':
    case 'unembed':
      return kind === loc.part
    case 'stage':
      return kind === 'stage' && a === loc.name
    case 'band': {
      if (kind !== 'depth') return false
      const [lo, hi] = loc.depth ?? [0, 100]
      const mid = ((L + 0.5) / DEPTH_ROWS) * 100
      return mid >= lo && mid <= hi
    }
  }
}

function describe(m: ModelSpec, key: string): string {
  const [kind, a, b] = key.split(':')
  if (kind === 'head') return `${m.name} · attention head ${a}.${b}`
  if (kind === 'attn') return `${m.name} · attention, layer ${a}`
  if (kind === 'mlp') return `${m.name} · MLP, layer ${a}`
  if (kind === 'resid') return `${m.name} · residual stream after block ${a}`
  if (kind === 'stage') return `${m.name} · ${a}`
  if (kind === 'depth') return `${m.name} · depth ${Number(a) * 5}–${(Number(a) + 1) * 5}%`
  return `${m.name} · ${kind === 'embed' ? 'embedding' : 'unembedding'}`
}

export default function ModelExplorer() {
  const { data } = useData('gpt2-map.json')
  const map = data as unknown as MapData | undefined
  const [modelId, setModelId] = useState('gpt2-small')
  const [sel, setSel] = useState<string | null>(null)
  const [hover, setHover] = useState<string | null>(null)
  const [overlay, setOverlay] = useState<Overlay>('findings')
  const [prompt, setPrompt] = useState<'repeat' | 'ioi'>('repeat')
  const m = modelById[modelId]

  // All locations: declared in entries + measured GPT-2 induction / previous-token heads.
  const hits: Hit[] = useMemo(() => {
    const out: Hit[] = []
    for (const e of entries) for (const loc of e.locations ?? []) out.push({ loc, entry: e })
    const ind = entries.find((e) => e.slug === 'induction-heads')
    if (map && ind)
      map.scores.induction.forEach((row, l) =>
        row.forEach((v, h) => {
          if (v > 0.4) out.push({ loc: { model: 'gpt2-small', part: 'head', layer: l, head: h, label: `induction head (measured score ${v.toFixed(2)})` }, entry: ind })
          if (map.scores.prev[l][h] > 0.4)
            out.push({ loc: { model: 'gpt2-small', part: 'head', layer: l, head: h, label: `previous-token head (measured score ${map.scores.prev[l][h].toFixed(2)})` }, entry: ind })
        }),
      )
    return out
  }, [map])
  const modelHits = hits.filter((h) => h.loc.model === modelId)
  const at = (key: string) => modelHits.filter((h) => covers(h.loc, key))
  const counts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const h of hits) c[h.loc.model] = (c[h.loc.model] ?? 0) + 1
    return c
  }, [hits])
  const color = (key: string) => {
    const hs = at(key)
    return hs.length ? CATEGORY[hs[0].entry.category].color : null
  }
  const scoreScale = useMemo(() => seqScale(1), [])
  const headFill = (l: number, h: number) => {
    if (overlay !== 'findings' && map && modelId === 'gpt2-small') return scoreScale(map.scores[overlay][l][h])
    return color(`head:${l}:${h}`) ?? 'var(--panel-2)'
  }
  const pick = (key: string) => setSel((s) => (s === key ? null : key))
  const focus = sel ?? hover
  const selHits = sel ? at(sel) : []
  const byEntry = useMemo(() => {
    const g = new Map<string, { entry: Entry; labels: Set<string> }>()
    for (const h of modelHits) {
      if (!g.has(h.entry.slug)) g.set(h.entry.slug, { entry: h.entry, labels: new Set() })
      g.get(h.entry.slug)!.labels.add(h.loc.label)
    }
    return [...g.values()]
  }, [modelHits])

  return (
    <section className="panel" aria-label="Model explorer">
      <h2 style={{ marginBottom: 4 }}>Explore the models: click a part to see what lives there</h2>
      <p className="viz-note" style={{ marginTop: 0 }}>
        Architecture diagrams of the models in this atlas. Highlighted parts host a circuit, manifold, neuron family,
        feature or direction (colored by category); click one to open its findings. Closed models are drawn by relative
        depth because their layer counts are not public.
      </p>
      <div className="tabs" role="tablist" aria-label="Choose a model">
        {MODELS.map((x) => (
          <button
            key={x.id}
            className={x.id === modelId ? 'on' : ''}
            onClick={() => {
              setModelId(x.id)
              setSel(null)
            }}
          >
            {x.name}
            {counts[x.id] ? ` · ${counts[x.id]}` : ''}
          </button>
        ))}
      </div>
      <div className="two-col">
        <div>
          {modelId === 'gpt2-small' && (
            <div className="viz-controls">
              color heads by
              <select className="sel" value={overlay} onChange={(e) => setOverlay(e.target.value as Overlay)}>
                <option value="findings">findings (category)</option>
                <option value="induction">measured induction score</option>
                <option value="prev">measured previous-token score</option>
                <option value="duplicate">measured duplicate-token score</option>
              </select>
            </div>
          )}
          {m.kind === 'transformer' ? (
            <TransformerDiagram m={m} headFill={headFill} color={color} focus={focus} onPick={pick} onHover={setHover} />
          ) : (
            <StageDiagram m={m} color={color} focus={focus} onPick={pick} onHover={setHover} />
          )}
          <div className="viz-controls">
            {Object.entries(CATEGORY).map(([k, c]) => (
              <span key={k} className="tag" style={{ background: c.color, color: '#fff' }}>
                {c.label}
              </span>
            ))}
          </div>
        </div>
        <div>
          {sel ? (
            <>
              <h3 style={{ marginTop: 0 }}>{describe(m, sel)}</h3>
              {selHits.length === 0 && <p className="viz-note">No finding in this atlas is localized here (yet).</p>}
              <ul className="repl-list">
                {selHits.map((h, i) => (
                  <li key={i}>
                    <CategoryTag c={h.entry.category} /> <FidelityBadge f={h.entry.fidelity} />{' '}
                    <Link to={`/entry/${h.entry.slug}`}>
                      <b>{h.entry.title}</b>
                    </Link>{' '}
                    — {h.loc.label}
                  </li>
                ))}
              </ul>
              {modelId === 'gpt2-small' && sel.startsWith('head:') && map && (
                <GptHeadPanel map={map} layer={+sel.split(':')[1]} head={+sel.split(':')[2]} prompt={prompt} setPrompt={setPrompt} />
              )}
              <button className="chip" onClick={() => setSel(null)} style={{ marginTop: 8 }}>
                ← back to all findings in {m.name}
              </button>
            </>
          ) : (
            <>
              <h3 style={{ marginTop: 0 }}>
                {m.name} <span className="tag">{m.open ? 'open weights' : 'closed'}</span> <span className="tag">{m.params}</span>
              </h3>
              <p className="viz-note">
                {m.blurb}{' '}
                <a href={m.source} target="_blank" rel="noopener">
                  source
                </a>
              </p>
              {byEntry.length === 0 ? (
                <p className="viz-note">Entries for this model are coming in a later batch.</p>
              ) : (
                <ul className="repl-list">
                  {byEntry.map(({ entry, labels }) => (
                    <li key={entry.slug}>
                      <CategoryTag c={entry.category} />{' '}
                      <Link to={`/entry/${entry.slug}`}>
                        <b>{entry.title}</b>
                      </Link>
                      <div className="viz-note">{[...labels].slice(0, 4).join(' · ')}{labels.size > 4 ? ` · +${labels.size - 4} more` : ''}</div>
                    </li>
                  ))}
                </ul>
              )}
              {focus && <p className="viz-note">Hovering: {describe(m, focus)} — click to open.</p>}
            </>
          )}
        </div>
      </div>
    </section>
  )
}

function GptHeadPanel({ map, layer, head, prompt, setPrompt }: { map: MapData; layer: number; head: number; prompt: 'repeat' | 'ioi'; setPrompt: (p: 'repeat' | 'ioi') => void }) {
  const pat = map.patterns[prompt]
  const color = useMemo(() => seqScale(1), [])
  const lab = (t: string, n: number) => (t === '<|endoftext|>' ? 'BOS' : t.trim().slice(0, n))
  return (
    <div>
      <p className="viz-note">
        Measured on repeated random tokens: induction {map.scores.induction[layer][head].toFixed(2)} · previous-token{' '}
        {map.scores.prev[layer][head].toFixed(2)} · duplicate-token {map.scores.duplicate[layer][head].toFixed(2)}
      </p>
      <div className="viz-controls">
        attention pattern on
        <select className="sel" value={prompt} onChange={(e) => setPrompt(e.target.value as 'repeat' | 'ioi')}>
          <option value="repeat">a repeated sentence</option>
          <option value="ioi">an IOI sentence</option>
        </select>
      </div>
      <Heatmap
        values={pat.pattern[layer][head]}
        color={color}
        cell={Math.floor(340 / pat.tokens.length)}
        yLabels={pat.tokens.map((t) => lab(t, 7))}
        xLabels={pat.tokens.map((t) => lab(t, 4))}
        xTitle="attends to"
        yTitle="from"
        maxLabelEvery={1}
        tooltip={(r, c, v) => `“${pat.tokens[r]}” → “${pat.tokens[c]}”: ${v.toFixed(2)}`}
      />
    </div>
  )
}

interface DiagramProps {
  m: ModelSpec
  color: (key: string) => string | null
  focus: string | null
  onPick: (key: string) => void
  onHover: (key: string | null) => void
}

/** Vertical transformer: embedding at the bottom, blocks stacked upward, unembedding on top, residual spine on the left. */
function TransformerDiagram({ m, headFill, color, focus, onPick, onHover }: DiagramProps & { headFill: (l: number, h: number) => string }) {
  const closed = !!m.relativeDepth
  const n = closed ? DEPTH_ROWS : m.layers!
  const W = 560
  const rowH = Math.max(13, Math.min(40, 520 / (n + 2)))
  const top = 30
  const H = top + (n + 2) * rowH + 10
  const yOf = (l: number) => top + (n - l) * rowH // row l (0 = first block) drawn bottom-up
  const spineX = 62
  const attnX = 84
  const attnW = m.attnOnly ? 440 : closed || !m.heads ? 280 : 300
  const mlpX = attnX + attnW + 14
  const mlpW = W - mlpX - 10
  const ring = (key: string) => (focus === key ? { stroke: '#f59e0b', strokeWidth: 2.5 } : {})
  const glow = (key: string) => (color(key) ? { filter: 'drop-shadow(0 0 3px rgba(255,255,255,0.35))' } : {})
  const handlers = (key: string) => ({
    onClick: () => onPick(key),
    onMouseEnter: () => onHover(key),
    onMouseLeave: () => onHover(null),
    style: { cursor: 'pointer', ...glow(key) },
  })
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img" aria-label={`${m.name} architecture`}>
      <text x={W / 2} y={16} textAnchor="middle" fontSize={12} fill="currentColor" opacity={0.75}>
        {closed ? 'output (logits)' : 'unembedding → next-token logits'}
      </text>
      {!closed && (
        <rect x={attnX} y={top - 6} width={W - attnX - 10} height={rowH - 6} rx={6} fill={color('unembed') ?? 'var(--panel-2)'} {...ring('unembed')} {...handlers('unembed')}>
          <title>unembedding</title>
        </rect>
      )}
      <line x1={spineX} x2={spineX} y1={top} y2={yOf(-1) + rowH / 2} stroke="currentColor" strokeWidth={3} opacity={0.35} />
      {[...Array(n).keys()].map((l) => {
        const y = yOf(l)
        if (closed) {
          const key = `depth:${l}`
          return (
            <g key={l}>
              <text x={spineX - 10} y={y + rowH / 2} fontSize={10} textAnchor="end" dominantBaseline="middle" fill="currentColor" opacity={0.7}>
                {l % 2 === 0 ? `${l * 5}%` : ''}
              </text>
              <rect x={attnX} y={y + 2} width={W - attnX - 10} height={rowH - 4} rx={4} fill={color(key) ?? 'var(--panel-2)'} opacity={color(key) ? 0.85 : 1} {...ring(key)} {...handlers(key)}>
                <title>depth {l * 5}–{(l + 1) * 5}% (layer count undisclosed)</title>
              </rect>
            </g>
          )
        }
        const resKey = `resid:${l}`
        return (
          <g key={l}>
            <text x={spineX - 14} y={y + rowH / 2} fontSize={Math.min(11, rowH - 2)} textAnchor="end" dominantBaseline="middle" fill="currentColor" opacity={0.75}>
              L{l}
            </text>
            {/* Attention: one cell per head (or a single block if heads are not drawn) */}
            {m.heads ? (
              [...Array(m.heads).keys()].map((h) => {
                const cw = (attnW - 4) / m.heads!
                const key = `head:${l}:${h}`
                return (
                  <rect key={h} x={attnX + 2 + h * cw} y={y + 3} width={Math.max(1, cw - 1.5)} height={rowH - 6} rx={2} fill={headFill(l, h)} {...ring(key)} {...handlers(key)}>
                    <title>
                      head {l}.{h}
                    </title>
                  </rect>
                )
              })
            ) : (
              <rect x={attnX} y={y + 3} width={attnW} height={rowH - 6} rx={4} fill={color(`attn:${l}`) ?? 'var(--panel-2)'} {...ring(`attn:${l}`)} {...handlers(`attn:${l}`)}>
                <title>block {l} (sequence mixing)</title>
              </rect>
            )}
            {!m.attnOnly && (
              <rect x={mlpX} y={y + 3} width={mlpW} height={rowH - 6} rx={4} fill={color(`mlp:${l}`) ?? 'var(--panel-2)'} {...ring(`mlp:${l}`)} {...handlers(`mlp:${l}`)}>
                <title>MLP {l}</title>
              </rect>
            )}
            {!m.attnOnly && rowH > 16 && l === n - 1 && (
              <text x={mlpX + mlpW / 2} y={y - 2} fontSize={10} textAnchor="middle" fill="currentColor" opacity={0.6}>
                MLP
              </text>
            )}
            {rowH > 16 && l === n - 1 && (
              <text x={attnX + attnW / 2} y={y - 2} fontSize={10} textAnchor="middle" fill="currentColor" opacity={0.6}>
                {m.heads ? `attention heads (${m.heads})` : 'blocks'}
              </text>
            )}
            {/* Residual stream node after this block */}
            <circle cx={spineX} cy={y + rowH / 2} r={Math.min(7, rowH / 2 - 1)} fill={color(resKey) ?? 'var(--panel)'} stroke="currentColor" strokeOpacity={0.5} {...ring(resKey)} {...handlers(resKey)}>
              <title>residual stream after block {l}</title>
            </circle>
          </g>
        )
      })}
      {!closed && (
        <g>
          <rect x={attnX} y={yOf(-1) + 3} width={W - attnX - 10} height={rowH - 6} rx={6} fill={color('embed') ?? 'var(--panel-2)'} {...ring('embed')} {...handlers('embed')}>
            <title>token (+ position) embedding</title>
          </rect>
          <text x={attnX + (W - attnX - 10) / 2} y={yOf(-1) + rowH / 2} textAnchor="middle" dominantBaseline="middle" fontSize={11} fill="currentColor" pointerEvents="none">
            embedding
          </text>
        </g>
      )}
      <text x={spineX} y={H - 2} textAnchor="middle" fontSize={10} fill="currentColor" opacity={0.6}>
        residual
      </text>
    </svg>
  )
}

/** Left-to-right stages (CNN layers or pipeline modules); box height ∝ √channels. */
function StageDiagram({ m, color, focus, onPick, onHover }: DiagramProps) {
  const stages = m.stages!
  const W = 560
  const gap = 6
  const bw = (W - 20 - gap * (stages.length - 1)) / stages.length
  const maxC = Math.max(...stages.map((s) => s.channels))
  const H = 250
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img" aria-label={`${m.name} architecture`}>
      {stages.map((s, i) => {
        const key = `stage:${s.name}`
        const h = 30 + 150 * Math.sqrt(s.channels / maxC)
        const x = 10 + i * (bw + gap)
        const y = 120 - h / 2
        const c = color(key)
        return (
          <g key={s.name} onClick={() => onPick(key)} onMouseEnter={() => onHover(key)} onMouseLeave={() => onHover(null)} style={{ cursor: 'pointer' }}>
            {i > 0 && <line x1={x - gap} x2={x} y1={120} y2={120} stroke="currentColor" opacity={0.4} strokeWidth={2} />}
            <rect x={x} y={y} width={bw} height={h} rx={6} fill={c ?? 'var(--panel-2)'} stroke={focus === key ? '#f59e0b' : 'currentColor'} strokeOpacity={focus === key ? 1 : 0.25} strokeWidth={focus === key ? 2.5 : 1}>
              <title>
                {s.name} ({s.channels} {m.kind === 'cnn' ? 'channels' : 'dims'})
              </title>
            </rect>
            <text x={x + bw / 2} y={210} fontSize={10} textAnchor="end" fill="currentColor" transform={`rotate(-40 ${x + bw / 2} 210)`}>
              {s.name}
            </text>
          </g>
        )
      })}
      <text x={W / 2} y={16} textAnchor="middle" fontSize={12} fill="currentColor" opacity={0.75}>
        input → output (box height ∝ √width of the layer)
      </text>
    </svg>
  )
}
