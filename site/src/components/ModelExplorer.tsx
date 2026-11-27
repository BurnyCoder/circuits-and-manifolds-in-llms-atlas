// Model Explorer: clickable, animated architecture diagrams of every model in the atlas.
// - Highlighted heads / MLPs / residual-stream positions / CNN stages / depth bands host a finding (category color).
// - Tiny thumbnails of each finding's mechanism are pinned next to where it lives; click one to EXPAND the entry's
//   full interactive visualization right here (or open its page).
// - "General LLM" / "General transformer" aggregate all models onto one generic block stack by relative depth;
//   "General neural network" puts every finding of every model (transformers, CNNs, autoencoders, world models) on one
//   generic input → hidden layers → output network.
// - GPT-2 small: heads can be colored by scores measured in this repo; a clicked head shows its real attention pattern.
// Global context: locations come from entry frontmatter (`locations`, src/lib/types.ts) plus measured induction /
// previous-token heads from public/data/gpt2-map.json; thumbnails from src/viz/thumbs.tsx; full visualizations from
// src/viz/registry.ts.
import { Suspense, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { entries } from '../lib/content'
import { useData } from '../lib/data'
import { CATEGORY } from '../lib/labels'
import { MODELS, modelById, type Family, type ModelSpec } from '../lib/models'
import type { Entry, Location } from '../lib/types'
import Heatmap from '../viz/primitives/Heatmap'
import { seqScale } from '../viz/primitives/common'
import { vizRegistry } from '../viz/registry'
import Thumb from '../viz/thumbs'
import { CategoryTag, FidelityBadge } from './Badges'

interface MapData {
  scores: { prev: number[][]; induction: number[][]; duplicate: number[][] }
  patterns: Record<string, { tokens: string[]; pattern: number[][][][] }>
}
type Overlay = 'findings' | 'induction' | 'prev' | 'duplicate'

const DEPTH_ROWS = 20 // closed + aggregate views: rows = 5% depth slices

// ---------- locations -> drawn components ----------
type VPart = 'vattn' | 'vmlp' | 'vresid' | 'vlayer' | 'embed' | 'unembed'
interface VLoc {
  part: VPart
  depth: [number, number]
}
interface Hit {
  loc: Location
  entry: Entry
  v?: VLoc
}

/** Aggregate views: place a location of a concrete model on the generic stack by relative depth (0–100%). */
function toVirtual(loc: Location, spec: ModelSpec): VLoc | null {
  if (loc.part === 'embed' || loc.part === 'unembed') return { part: loc.part, depth: [0, 100] }
  if (loc.part === 'band') return { part: 'vresid', depth: loc.depth ?? [0, 100] }
  if (loc.part === 'stage' && spec.stages) {
    // CNN / pipeline stage i of n covers relative depth [i/n, (i+1)/n] (input stage first).
    const i = spec.stages.findIndex((st) => st.name === loc.name)
    return i < 0 ? null : { part: 'vresid', depth: [(i / spec.stages.length) * 100, ((i + 1) / spec.stages.length) * 100] }
  }
  if (!spec.layers) return null
  const n = spec.layers
  const span = (lo: number, hi: number): [number, number] => [(lo / n) * 100, ((hi + 1) / n) * 100]
  if (loc.part === 'head' || loc.part === 'attn') return { part: 'vattn', depth: span(loc.layer!, loc.layer!) }
  if (loc.part === 'mlp') return { part: 'vmlp', depth: span(loc.layer!, loc.layer!) }
  if (loc.part === 'resid') {
    const [lo, hi] = loc.layers ?? [loc.layer!, loc.layer!]
    return { part: 'vresid', depth: span(lo, hi) }
  }
  return null
}

function coversVirtual(v: VLoc, key: string): boolean {
  const [kind, a] = key.split(':')
  if (kind === 'embed' || kind === 'unembed') return v.part === kind
  if (kind !== v.part) return false
  const lo = Number(a) * (100 / DEPTH_ROWS)
  return v.depth[0] < lo + 100 / DEPTH_ROWS && v.depth[1] > lo
}

/** Keys: head:L:H, attn:L, mlp:L, resid:L, embed, unembed, stage:NAME, depth:I, vattn:I, vmlp:I, vresid:I, vlayer:I */
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

/** One representative drawn component per hit (where its thumbnail pin points). */
function anchorKey(h: Hit): string {
  if (h.v) {
    const mid = (h.v.depth[0] + h.v.depth[1]) / 2
    const row = Math.min(DEPTH_ROWS - 1, Math.floor(mid / (100 / DEPTH_ROWS)))
    return h.v.part === 'embed' || h.v.part === 'unembed' ? h.v.part : `${h.v.part}:${row}`
  }
  const l = h.loc
  switch (l.part) {
    case 'head':
      return `head:${l.layer}:${l.head}`
    case 'attn':
    case 'mlp':
      return `${l.part}:${l.layer}`
    case 'resid':
      return `resid:${l.layers ? Math.round((l.layers[0] + l.layers[1]) / 2) : l.layer}`
    case 'stage':
      return `stage:${l.name}`
    case 'band': {
      const [lo, hi] = l.depth ?? [0, 100]
      return `depth:${Math.min(DEPTH_ROWS - 1, Math.floor((lo + hi) / 2 / 5))}`
    }
    default:
      return l.part
  }
}

function describe(m: ModelSpec, key: string): string {
  const [kind, a, b] = key.split(':')
  const where = `at ${Number(a) * 5}–${(Number(a) + 1) * 5}% depth`
  switch (kind) {
    case 'head':
      return `${m.name} · attention head ${a}.${b}`
    case 'attn':
      return `${m.name} · attention, layer ${a}`
    case 'mlp':
      return `${m.name} · MLP, layer ${a}`
    case 'resid':
      return `${m.name} · residual stream after block ${a}`
    case 'stage':
      return `${m.name} · ${a}`
    case 'depth':
      return `${m.name} · ${where}`
    case 'vattn':
      return `Generic transformer · attention ${where}`
    case 'vmlp':
      return `Generic transformer · MLP ${where}`
    case 'vresid':
      return `Generic transformer · residual stream ${where}`
    case 'vlayer':
      return `Generic neural network · hidden layer ${where}`
    default:
      if (m.kind === 'network') return `Generic neural network · ${kind === 'embed' ? 'input layer' : 'output layer'}`
      return `${m.name} · ${kind === 'embed' ? 'embedding' : 'unembedding'}`
  }
}

// ---------- main component ----------
export default function ModelExplorer() {
  const { data } = useData('gpt2-map.json')
  const map = data as unknown as MapData | undefined
  const [modelId, setModelId] = useState('general-nn')
  const [sel, setSel] = useState<string | null>(null)
  const [hover, setHover] = useState<string | null>(null)
  const [overlay, setOverlay] = useState<Overlay>('findings')
  const [prompt, setPrompt] = useState<'repeat' | 'ioi'>('repeat')
  const [expanded, setExpanded] = useState<string | null>(null)
  const m = modelById[modelId]

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

  const modelHits: Hit[] = useMemo(() => {
    if (!m.aggregate) return hits.filter((h) => h.loc.model === modelId)
    const network = m.kind === 'network'
    const out: Hit[] = []
    for (const h of hits) {
      const spec = modelById[h.loc.model]
      if (!spec?.family || !m.aggregate.includes(spec.family)) continue
      const v = toVirtual(h.loc, spec)
      // The generic network has one kind of hidden layer: attention / MLP / residual / stage all collapse onto it.
      if (v) out.push({ ...h, v: network && v.part !== 'embed' && v.part !== 'unembed' ? { ...v, part: 'vlayer' } : v, loc: { ...h.loc, label: `${spec.name}: ${h.loc.label}` } })
    }
    if (network)
      for (const e of entries.filter((x) => !x.locations?.length))
        out.push({
          entry: e,
          v: { part: 'vlayer', depth: [47.5, 52.5] },
          loc: { model: 'none', part: 'band', depth: [47.5, 52.5], label: `${e.models.join(', ')}: no specific layer (toy / synthetic setup or not localized in this atlas)` },
        })
    return out
  }, [hits, m, modelId])

  const at = (key: string) => modelHits.filter((h) => (h.v ? coversVirtual(h.v, key) : covers(h.loc, key)))
  const color = (key: string) => {
    const hs = at(key)
    return hs.length ? CATEGORY[hs[0].entry.category].color : null
  }
  /** Category colors of every finding at a component, grouped by category (for the multi-colored network rows). */
  const mix = (key: string) => {
    const order = Object.keys(CATEGORY)
    return at(key)
      .map((h) => h.entry.category)
      .sort((a, b) => order.indexOf(a) - order.indexOf(b))
      .map((c) => CATEGORY[c].color)
  }
  const maxHits = useMemo(() => {
    if (!m.aggregate) return 1
    let mx = 1
    for (const part of ['vattn', 'vmlp', 'vresid', 'vlayer'])
      for (let i = 0; i < DEPTH_ROWS; i++) mx = Math.max(mx, modelHits.filter((h) => h.v && coversVirtual(h.v, `${part}:${i}`)).length)
    return mx
  }, [m, modelHits])
  const intensity = (key: string) => (m.aggregate ? at(key).length / maxHits : 1)
  const counts = useMemo(() => {
    const c: Record<string, number> = {}
    for (const h of hits) {
      c[h.loc.model] = (c[h.loc.model] ?? 0) + 1
      const fam = modelById[h.loc.model]?.family
      for (const agg of MODELS.filter((x) => x.aggregate)) if (fam && agg.aggregate!.includes(fam as Family)) c[agg.id] = (c[agg.id] ?? 0) + 1
    }
    c['general-nn'] = (c['general-nn'] ?? 0) + entries.filter((e) => !e.locations?.length).length // unlocalized entries
    return c
  }, [hits])
  const scoreScale = useMemo(() => seqScale(1), [])
  const headFill = (l: number, h: number) => {
    if (overlay !== 'findings' && map && modelId === 'gpt2-small') return scoreScale(map.scores[overlay][l][h])
    return color(`head:${l}:${h}`)
  }
  // One pin per entry, anchored at the entry's first location in this model.
  const pins = useMemo(() => {
    const seen = new Map<string, { entry: Entry; key: string; n: number }>()
    for (const h of modelHits) {
      const p = seen.get(h.entry.slug)
      if (p) p.n++
      else seen.set(h.entry.slug, { entry: h.entry, key: anchorKey(h), n: 1 })
    }
    return [...seen.values()]
  }, [modelHits])
  const pick = (key: string) => setSel((s) => (s === key ? null : key))
  const expand = (slug: string) => setExpanded((s) => (s === slug ? null : slug))
  const focus = sel ?? hover
  const selHits = sel ? at(sel) : []
  const expandedEntry = expanded ? entries.find((e) => e.slug === expanded) : undefined

  return (
    <section className="panel explorer" aria-label="Model explorer">
      <h2 style={{ marginBottom: 4 }}>Explore the models — click a glowing part or a tiny circuit</h2>
      <p className="viz-note" style={{ marginTop: 0 }}>
        Architecture diagrams of the models in this atlas. Glowing parts host a circuit, manifold, neuron family, feature or
        direction (colored by category). The tiny pictures on the right are miniature versions of those findings — click one
        to expand its full interactive visualization below. Closed models are drawn by relative depth (layer counts are not
        public); the three <b>General</b> views pool models by relative depth — <b>General neural network</b> holds every finding in the atlas.
      </p>
      <div className="tabs model-tabs" role="tablist" aria-label="Choose a model">
        {MODELS.map((x) => (
          <button
            key={x.id}
            className={`${x.id === modelId ? 'on' : ''} ${x.aggregate ? 'agg' : ''}`}
            onClick={() => {
              setModelId(x.id)
              setSel(null)
            }}
          >
            {x.aggregate ? '✦ ' : x.open ? '' : '🔒 '}
            {x.name}
            {counts[x.id] ? <span className="count">{counts[x.id]}</span> : null}
          </button>
        ))}
      </div>
      <div className="explorer-grid">
        <div style={{ minWidth: 0 }}>
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
          {m.kind === 'network' ? (
            <NetworkDiagram m={m} color={color} mix={mix} intensity={intensity} focus={focus} onPick={pick} onHover={setHover} pins={pins} expanded={expanded} onExpand={expand} />
          ) : m.kind === 'transformer' ? (
            <TransformerDiagram m={m} headFill={headFill} color={color} intensity={intensity} focus={focus} onPick={pick} onHover={setHover} pins={pins} expanded={expanded} onExpand={expand} />
          ) : (
            <StageDiagram m={m} color={color} focus={focus} onPick={pick} onHover={setHover} pins={pins} expanded={expanded} onExpand={expand} />
          )}
          {m.aggregate && <p className="viz-note">Shading = number of findings at that depth (darker = more). Click a slice to list them with their source models.</p>}
          <div className="viz-controls">
            {Object.entries(CATEGORY).map(([k, c]) => (
              <span key={k} className="tag" style={{ background: c.color, color: '#fff' }}>
                {c.label}
              </span>
            ))}
          </div>
        </div>
        <aside style={{ minWidth: 0 }}>
          {sel ? (
            <>
              <h3 style={{ marginTop: 0 }}>{describe(m, sel)}</h3>
              {selHits.length === 0 && <p className="viz-note">No finding in this atlas is localized here (yet).</p>}
              <FindingList hits={selHits} expanded={expanded} onExpand={expand} />
              {modelId === 'gpt2-small' && sel.startsWith('head:') && map && <GptHeadPanel map={map} layer={+sel.split(':')[1]} head={+sel.split(':')[2]} prompt={prompt} setPrompt={setPrompt} />}
              <button className="chip" onClick={() => setSel(null)} style={{ marginTop: 8 }}>
                ← all findings in {m.name}
              </button>
            </>
          ) : (
            <>
              <h3 style={{ marginTop: 0 }}>
                {m.name} <span className="tag">{m.aggregate ? 'aggregate' : m.open ? 'open weights' : 'closed'}</span> <span className="tag">{m.params}</span>
              </h3>
              <p className="viz-note">
                {m.blurb}{' '}
                <a href={m.source} target="_blank" rel="noopener">
                  source
                </a>
              </p>
              {pins.length === 0 ? (
                <p className="viz-note">Entries for this model are coming in a later batch.</p>
              ) : (
                <FindingList hits={pins.map((p) => ({ entry: p.entry, loc: { model: modelId, part: 'embed', label: `${p.n} location${p.n > 1 ? 's' : ''} in this model` } }))} expanded={expanded} onExpand={expand} />
              )}
              {focus && <p className="viz-note">Hovering: {describe(m, focus)} — click to open.</p>}
            </>
          )}
        </aside>
      </div>
      {expandedEntry && <ExpandedViz entry={expandedEntry} onClose={() => setExpanded(null)} />}
    </section>
  )
}

function FindingList({ hits, expanded, onExpand }: { hits: Hit[]; expanded: string | null; onExpand: (slug: string) => void }) {
  return (
    <ul className="finding-list">
      {hits.map((h, i) => (
        <li key={i} className={expanded === h.entry.slug ? 'on' : ''}>
          <button className="thumb-btn" onClick={() => onExpand(h.entry.slug)} title="Expand this visualization">
            <Thumb slug={h.entry.slug} category={h.entry.category} title={h.entry.title} />
          </button>
          <div>
            <div>
              <CategoryTag c={h.entry.category} /> <FidelityBadge f={h.entry.fidelity} />
            </div>
            <Link to={`/entry/${h.entry.slug}`}>
              <b>{h.entry.title}</b>
            </Link>
            <div className="viz-note">{h.loc.label}</div>
          </div>
        </li>
      ))}
    </ul>
  )
}

/** The full interactive visualization of an entry, embedded in the explorer. */
function ExpandedViz({ entry, onClose }: { entry: Entry; onClose: () => void }) {
  const { data, error } = useData(entry.data)
  const Viz = entry.viz ? vizRegistry[entry.viz] : undefined
  return (
    <div className="expanded" style={{ ['--cat-color' as string]: CATEGORY[entry.category].color }}>
      <div className="expanded-head">
        <div style={{ width: 84, flex: 'none' }}>
          <Thumb slug={entry.slug} category={entry.category} title={entry.title} />
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <h3 style={{ margin: 0 }}>{entry.title}</h3>
          <p className="viz-note" style={{ margin: '2px 0' }}>
            {entry.summary}
          </p>
          <Link to={`/entry/${entry.slug}`}>Open the full entry (explanations, sources, replications) →</Link>
        </div>
        <button className="chip" onClick={onClose} aria-label="Close">
          ✕ close
        </button>
      </div>
      {!Viz ? (
        <p className="viz-note">This entry is a write-up without an interactive visualization yet.</p>
      ) : error ? (
        entry.status === 'pending' ? (
          <p className="viz-note">⏳ Results pending: this entry's full run has not finished yet.</p>
        ) : (
          <p className="error">{error}</p>
        )
      ) : entry.data && !data ? (
        <p>Loading data…</p>
      ) : (
        <Suspense fallback={<p>Loading visualization…</p>}>
          <Viz data={data} entry={entry} />
        </Suspense>
      )}
    </div>
  )
}

function GptHeadPanel({ map, layer, head, prompt, setPrompt }: { map: MapData; layer: number; head: number; prompt: 'repeat' | 'ioi'; setPrompt: (p: 'repeat' | 'ioi') => void }) {
  const pat = map.patterns[prompt]
  const color = useMemo(() => seqScale(1), [])
  const lab = (t: string, n: number) => (t === '<|endoftext|>' ? 'BOS' : t.trim().slice(0, n))
  return (
    <div>
      <p className="viz-note">
        Measured on repeated random tokens: induction {map.scores.induction[layer][head].toFixed(2)} · previous-token {map.scores.prev[layer][head].toFixed(2)} · duplicate-token{' '}
        {map.scores.duplicate[layer][head].toFixed(2)}
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

// ---------- diagrams ----------
interface Pin {
  entry: Entry
  key: string
  n: number
}
interface DiagramProps {
  m: ModelSpec
  color: (key: string) => string | null
  focus: string | null
  onPick: (key: string) => void
  onHover: (key: string | null) => void
  pins: Pin[]
  expanded: string | null
  onExpand: (slug: string) => void
}

const PIN_W = 96
const PIN_H = 64
const PIN_GAP = 22

const PIN_STEP = PIN_H + PIN_GAP
const MAX_PIN_COLS = 3

/** Columns and total height needed so that `n` pins fit in [top, bottom] (height grows when 3 columns are not enough). */
function pinGrid(n: number, top: number, bottom: number): { cols: number; bottom: number } {
  const slots = Math.max(1, Math.floor((bottom - top) / PIN_STEP))
  const cols = Math.min(MAX_PIN_COLS, Math.max(1, Math.ceil(n / slots)))
  return { cols, bottom: Math.max(bottom, top + Math.ceil(n / cols) * PIN_STEP) }
}

/**
 * Lay out pins in `cols` columns to the right of the diagram, each as close as possible to its anchor's y, never
 * overlapping and never leaving [top, bottom]. Pins (sorted by anchor y) are dealt round-robin into the columns so every
 * column spans the full height; within a column a forward pass removes overlaps and a backward pass pulls pins back
 * above `bottom` (classic 1-D label placement). Requires ceil(n / cols) · PIN_STEP ≤ bottom − top (see pinGrid).
 */
function layoutPins<T extends Pin & { y: number }>(pins: T[], top: number, bottom: number, x0: number, cols: number) {
  const sorted = [...pins].sort((a, b) => a.y - b.y)
  const out: (T & { px: number; py: number })[] = []
  for (let c = 0; c < cols; c++) {
    const col = sorted.filter((_, i) => i % cols === c)
    // A card is PIN_STEP tall (thumbnail + title strip), so its top may go no lower than bottom − PIN_STEP.
    const ys = col.map((p) => Math.min(Math.max(p.y - PIN_H / 2, top), bottom - PIN_STEP))
    for (let i = 1; i < ys.length; i++) ys[i] = Math.max(ys[i], ys[i - 1] + PIN_STEP) // push down to remove overlaps
    for (let i = ys.length - 1; i >= 0; i--) ys[i] = Math.min(ys[i], i === ys.length - 1 ? bottom - PIN_STEP : ys[i + 1] - PIN_STEP) // pull up into range
    col.forEach((p, i) => out.push({ ...p, px: x0 + c * (PIN_W + 12), py: ys[i] }))
  }
  return out
}

function PinCard({ p, x, y, active, onExpand, onHover, anchorX, anchorY }: { p: Pin; x: number; y: number; active: boolean; onExpand: (s: string) => void; onHover: (k: string | null) => void; anchorX: number; anchorY: number }) {
  const c = CATEGORY[p.entry.category].color
  return (
    <g className="pin" onClick={() => onExpand(p.entry.slug)} onMouseEnter={() => onHover(p.key)} onMouseLeave={() => onHover(null)} style={{ cursor: 'pointer' }}>
      <path
        d={`M${anchorX},${anchorY} C${(anchorX + x) / 2},${anchorY} ${(anchorX + x) / 2},${y + PIN_H / 2} ${x},${y + PIN_H / 2}`}
        fill="none"
        stroke={c}
        strokeWidth={active ? 2.5 : 1.4}
        strokeOpacity={active ? 1 : 0.6}
        strokeDasharray={active ? undefined : '4 3'}
      />
      <circle cx={anchorX} cy={anchorY} r={3} fill={c} />
      <foreignObject x={x} y={y} width={PIN_W} height={PIN_H + PIN_GAP}>
        <div className={`pin-card ${active ? 'on' : ''}`} style={{ borderColor: c }}>
          <div style={{ height: PIN_H - 2 }}>
            <Thumb slug={p.entry.slug} category={p.entry.category} title={p.entry.title} />
          </div>
          <div className="pin-title">{p.entry.title.replace(/\s*\(.*\)$/, '')}</div>
        </div>
      </foreignObject>
    </g>
  )
}

function Defs() {
  return (
    <defs>
      <linearGradient id="resid-grad" x1="0" y1="1" x2="0" y2="0">
        <stop offset="0" stopColor="#06b6d4" />
        <stop offset="0.5" stopColor="#8b5cf6" />
        <stop offset="1" stopColor="#ec4899" />
      </linearGradient>
      <filter id="glow" x="-40%" y="-40%" width="180%" height="180%">
        <feGaussianBlur stdDeviation="2.4" result="b" />
        <feMerge>
          <feMergeNode in="b" />
          <feMergeNode in="SourceGraphic" />
        </feMerge>
      </filter>
    </defs>
  )
}

/** Vertical transformer: tokens + embedding at the bottom, blocks stacked upward, unembedding/output on top. */
function TransformerDiagram({
  m,
  headFill,
  color,
  intensity,
  focus,
  onPick,
  onHover,
  pins,
  expanded,
  onExpand,
}: DiagramProps & { headFill: (l: number, h: number) => string | null; intensity: (key: string) => number }) {
  const virtual = !!m.aggregate
  const closed = !!m.relativeDepth && !virtual
  const n = closed || virtual ? DEPTH_ROWS : m.layers!
  const D = 470 // diagram width (pins go to the right)
  const rowH = Math.max(14, Math.min(40, 540 / (n + 3)))
  const top = 46
  const yOf = (l: number) => top + (n - l) * rowH
  const bottom = yOf(-1) + rowH + 40
  const spineX = 52
  const attnX = 72
  const attnW = m.attnOnly ? D - attnX - 10 : 250
  const mlpX = attnX + attnW + 12
  const mlpW = D - mlpX - 10
  const grid = pinGrid(pins.length, top - 30, bottom)
  const cols = grid.cols
  const W = D + 30 + cols * (PIN_W + 12)
  const H = grid.bottom + 10 // taller than the diagram when the pins need more room
  const isOn = (key: string) => color(key) !== null
  const op = (key: string) => (isOn(key) ? 0.25 + 0.75 * intensity(key) : 1)
  const ring = (key: string) => (focus === key ? { stroke: '#f59e0b', strokeWidth: 2.5 } : {})
  const handlers = (key: string) => ({
    onClick: () => onPick(key),
    onMouseEnter: () => onHover(key),
    onMouseLeave: () => onHover(null),
    style: { cursor: 'pointer' },
    filter: isOn(key) ? 'url(#glow)' : undefined,
  })
  // Anchor coordinates for pins.
  const anchor = (key: string): [number, number] => {
    const [kind, a, b] = key.split(':')
    const L = Number(a)
    if (kind === 'unembed') return [D - 10, top - 14]
    if (kind === 'embed') return [D - 10, yOf(-1) + rowH / 2]
    const y = yOf(L) + rowH / 2
    if (kind === 'head' && m.heads) return [attnX + 2 + (Number(b) + 0.5) * ((attnW - 4) / m.heads), y]
    if (kind === 'attn' || kind === 'vattn') return [attnX + attnW, y]
    if (kind === 'mlp' || kind === 'vmlp') return [mlpX + mlpW, y]
    if (kind === 'depth') return [D - 10, y]
    return [spineX + 7, y] // residual
  }
  const placed = layoutPins(
    pins.map((p) => ({ ...p, y: anchor(p.key)[1] })),
    top - 30,
    grid.bottom,
    D + 30,
    cols,
  )
  const tokens = ['When', 'Mary', 'and', 'John', 'went', '…', 'to']
  const neuronDots = 9
  const tokW = (D - attnX - 10) / tokens.length
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img" aria-label={`${m.name} architecture`}>
      <Defs />
      <rect x={attnX} y={6} width={D - attnX - 10} height={24} rx={12} fill="url(#resid-grad)" opacity={0.18} />
      <text x={(attnX + D) / 2} y={22} textAnchor="middle" fontSize={12} fill="currentColor">
        {m.family === 'llm' || virtual ? 'next-token prediction  →  “ Mary”' : closed ? 'output' : 'output logits'}
      </text>
      {!closed && (
        <rect x={attnX} y={top - 14} width={D - attnX - 10} height={rowH - 8} rx={8} fill={color('unembed') ?? 'var(--panel-2)'} {...ring('unembed')} {...handlers('unembed')}>
          <title>unembedding (residual → logits)</title>
        </rect>
      )}
      <line x1={spineX} x2={spineX} y1={top - 6} y2={yOf(-1) + rowH / 2} stroke="url(#resid-grad)" strokeWidth={7} strokeLinecap="round" opacity={0.85} />
      <line className="flow" x1={spineX} x2={spineX} y1={yOf(-1) + rowH / 2} y2={top - 6} stroke="#fff" strokeOpacity={0.55} strokeWidth={2.5} />
      {[...Array(n).keys()].map((l) => {
        const y = yOf(l)
        const label = closed || virtual ? (l % 2 === 0 ? `${l * 5}%` : '') : `L${l}`
        const readWrite = <line x1={spineX} x2={attnX} y1={y + rowH / 2} y2={y + rowH / 2} stroke="currentColor" strokeOpacity={0.15} />
        if (closed) {
          const key = `depth:${l}`
          return (
            <g key={l}>
              <text x={spineX - 12} y={y + rowH / 2} fontSize={10} textAnchor="end" dominantBaseline="middle" fill="currentColor" opacity={0.7}>
                {label}
              </text>
              {readWrite}
              <rect x={attnX} y={y + 2} width={D - attnX - 10} height={rowH - 4} rx={6} fill={color(key) ?? 'var(--panel-2)'} fillOpacity={op(key)} {...ring(key)} {...handlers(key)}>
                <title>
                  depth {l * 5}–{(l + 1) * 5}% (layer count undisclosed)
                </title>
              </rect>
            </g>
          )
        }
        const aKey = virtual ? `vattn:${l}` : `attn:${l}`
        const mKey = virtual ? `vmlp:${l}` : `mlp:${l}`
        const rKey = virtual ? `vresid:${l}` : `resid:${l}`
        const mlpOn = isOn(mKey)
        return (
          <g key={l}>
            {l % 2 === 0 && <rect x={attnX - 6} y={y + 1} width={D - attnX - 2} height={rowH - 2} rx={8} fill="var(--panel-2)" opacity={0.45} />}
            <text x={spineX - 14} y={y + rowH / 2} fontSize={Math.min(11, rowH - 3)} textAnchor="end" dominantBaseline="middle" fill="currentColor" opacity={0.75}>
              {label}
            </text>
            {readWrite}
            {m.heads && !virtual ? (
              [...Array(m.heads).keys()].map((h) => {
                const cw = (attnW - 4) / m.heads!
                const key = `head:${l}:${h}`
                const f = headFill(l, h)
                return (
                  <rect key={h} x={attnX + 2 + h * cw} y={y + 4} width={Math.max(1, cw - 2)} height={rowH - 8} rx={Math.min(5, cw / 3)} fill={f ?? 'var(--panel)'} stroke={f ? 'none' : 'currentColor'} strokeOpacity={0.18} {...ring(key)} {...handlers(key)}>
                    <title>
                      head {l}.{h}
                    </title>
                  </rect>
                )
              })
            ) : (
              <rect x={attnX} y={y + 3} width={attnW} height={rowH - 6} rx={6} fill={color(aKey) ?? 'var(--panel)'} fillOpacity={op(aKey)} stroke="currentColor" strokeOpacity={0.15} {...ring(aKey)} {...handlers(aKey)}>
                <title>{virtual ? `attention at ${l * 5}–${(l + 1) * 5}% depth` : `block ${l} (sequence mixing)`}</title>
              </rect>
            )}
            {!m.attnOnly && (
              <g {...handlers(mKey)}>
                <rect x={mlpX} y={y + 3} width={mlpW} height={rowH - 6} rx={6} fill={color(mKey) ?? 'var(--panel)'} fillOpacity={op(mKey)} stroke="currentColor" strokeOpacity={0.15} {...ring(mKey)}>
                  <title>{virtual ? `MLP at ${l * 5}–${(l + 1) * 5}% depth` : `MLP ${l}`}</title>
                </rect>
                {rowH > 18 &&
                  [...Array(neuronDots).keys()].map((k) => (
                    <circle key={k} cx={mlpX + ((k + 0.5) * mlpW) / neuronDots} cy={y + rowH / 2} r={Math.min(3, rowH / 8)} fill={mlpOn ? '#fff' : 'currentColor'} opacity={mlpOn ? 0.85 : 0.18} pointerEvents="none" />
                  ))}
              </g>
            )}
            {l === n - 1 && rowH > 15 && (
              <>
                <text x={attnX + attnW / 2} y={y - 3} fontSize={10} textAnchor="middle" fill="currentColor" opacity={0.6}>
                  {virtual ? 'attention (all heads)' : m.heads ? `attention heads (${m.heads})` : 'sequence-mixing blocks'}
                </text>
                {!m.attnOnly && (
                  <text x={mlpX + mlpW / 2} y={y - 3} fontSize={10} textAnchor="middle" fill="currentColor" opacity={0.6}>
                    MLP neurons
                  </text>
                )}
              </>
            )}
            <circle cx={spineX} cy={y + rowH / 2} r={Math.min(7, rowH / 2 - 1)} fill={color(rKey) ?? 'var(--panel)'} fillOpacity={op(rKey)} stroke="url(#resid-grad)" strokeWidth={2} {...ring(rKey)} {...handlers(rKey)}>
              <title>residual stream {virtual ? `at ${l * 5}–${(l + 1) * 5}% depth` : `after block ${l}`}</title>
            </circle>
          </g>
        )
      })}
      {!closed && (
        <g>
          <rect x={attnX} y={yOf(-1) + 3} width={D - attnX - 10} height={rowH - 6} rx={8} fill={color('embed') ?? 'var(--panel-2)'} {...ring('embed')} {...handlers('embed')}>
            <title>token (+ position) embedding</title>
          </rect>
          <text x={(attnX + D) / 2} y={yOf(-1) + rowH / 2 + 1} textAnchor="middle" dominantBaseline="middle" fontSize={11} fill="currentColor" pointerEvents="none">
            embedding
          </text>
        </g>
      )}
      {(m.family === 'llm' || virtual) &&
        tokens.map((t, i) => (
          <g key={i}>
            <rect x={attnX + i * tokW + 2} y={yOf(-1) + rowH + 6} width={tokW - 4} height={20} rx={6} fill="var(--panel)" stroke="url(#resid-grad)" />
            <text x={attnX + i * tokW + tokW / 2} y={yOf(-1) + rowH + 20} fontSize={10} textAnchor="middle" fill="currentColor">
              {t}
            </text>
          </g>
        ))}
      {placed.map((p) => {
        const [ax, ay] = anchor(p.key)
        return <PinCard key={p.entry.slug} p={p} x={p.px} y={p.py} active={expanded === p.entry.slug} onExpand={onExpand} onHover={onHover} anchorX={ax} anchorY={ay} />
      })}
    </svg>
  )
}

/**
 * Generic neural network for the "General neural network" view: input layer at the bottom, DEPTH_ROWS hidden layers
 * (5% of relative depth each), output layer on top. Each layer is a row of neurons linked to nearby neurons of the next
 * layer; rows that host findings glow in the first finding's category color (opacity = how many findings). Pins sit on
 * the right as in the transformer views.
 */
function NetworkDiagram({ m, color, mix, intensity, focus, onPick, onHover, pins, expanded, onExpand }: DiagramProps & { mix: (key: string) => string[]; intensity: (key: string) => number }) {
  const n = DEPTH_ROWS
  const D = 470
  const top = 50
  // Stretch the network to the height the pins need (≥ 26 px per layer), so the diagram and its pins line up.
  const rowH = Math.max(26, (Math.ceil(pins.length / MAX_PIN_COLS) * PIN_STEP - 52) / (n + 2))
  const yOf = (row: number) => top + (n - row) * rowH + rowH / 2 // row −1 = input, row n = output
  const bottom = yOf(-1) + rowH + 26
  const spineX = 40
  const x0 = 78
  const x1 = D - 18
  // Neurons per layer: narrow input/output, wider in the middle (the usual hourglass-free "bulge" picture).
  const width = (row: number) => (row === -1 ? 7 : row === n ? 5 : Math.round(8 + 3 * Math.sin(((row + 0.5) / n) * Math.PI)))
  const nx = (row: number, i: number) => {
    const k = width(row)
    const span = (x1 - x0) * (0.5 + (0.5 * k) / 11)
    return (x0 + x1) / 2 - span / 2 + (i * span) / (k - 1)
  }
  const keyOf = (row: number) => (row === -1 ? 'embed' : row === n ? 'unembed' : `vlayer:${row}`)
  const rows = Array.from({ length: n + 2 }, (_, i) => i - 1)
  const grid = pinGrid(pins.length, top - 34, bottom)
  const W = D + 30 + grid.cols * (PIN_W + 12)
  const H = grid.bottom + 10
  const anchorY = (key: string) => {
    const [kind, a] = key.split(':')
    return yOf(kind === 'embed' ? -1 : kind === 'unembed' ? n : Number(a))
  }
  const placed = layoutPins(
    pins.map((p) => ({ ...p, y: anchorY(p.key) })),
    top - 34,
    grid.bottom,
    D + 30,
    grid.cols,
  )
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img" aria-label={`${m.name} diagram`}>
      <Defs />
      <line className="flow" x1={spineX} x2={spineX} y1={yOf(-1)} y2={yOf(n)} stroke="url(#resid-grad)" strokeWidth={3} />
      {/* connections between consecutive layers: each neuron to the neurons at a similar relative position */}
      {rows.slice(0, -1).map((r) => {
        const k1 = width(r)
        const k2 = width(r + 1)
        const c = color(keyOf(r + 1)) ?? color(keyOf(r))
        const segs: [number, number, number, number][] = []
        for (let i = 0; i < k1; i++)
          for (let j = 0; j < k2; j++) if (Math.abs(i / (k1 - 1) - j / (k2 - 1)) <= 0.22) segs.push([nx(r, i), yOf(r), nx(r + 1, j), yOf(r + 1)])
        return (
          <g key={`e${r}`} stroke={c ?? 'currentColor'} strokeOpacity={c ? 0.35 : 0.12} strokeWidth={1}>
            {segs.map(([a, b, cc, d], i) => (
              <line key={i} x1={a} y1={b} x2={cc} y2={d} />
            ))}
          </g>
        )
      })}
      {rows.map((r) => {
        const key = keyOf(r)
        const c = color(key)
        const y = yOf(r)
        return (
          <g key={key} onClick={() => onPick(key)} onMouseEnter={() => onHover(key)} onMouseLeave={() => onHover(null)} style={{ cursor: 'pointer' }} filter={c ? 'url(#glow)' : undefined}>
            <rect x={x0 - 14} y={y - rowH / 2 + 2} width={x1 - x0 + 28} height={rowH - 4} rx={10} fill={focus === key ? '#f59e0b22' : 'transparent'} stroke={focus === key ? '#f59e0b' : 'none'} />
            {Array.from({ length: width(r) }, (_, i) => {
              // Neuron i takes the color of the finding at the same fraction along the row's category-sorted list,
              // so a row's colors show the mix of finding types at that depth.
              const cs = mix(key)
              const ci = cs.length ? cs[Math.min(cs.length - 1, Math.floor((i * cs.length) / width(r)))] : null
              return <circle key={i} cx={nx(r, i)} cy={y} r={r === -1 || r === n ? 6.5 : 5.5} fill={ci ?? 'var(--panel-2)'} fillOpacity={ci ? 0.4 + 0.6 * intensity(key) : 1} stroke="currentColor" strokeOpacity={ci ? 0 : 0.25} />
            })}
            <circle cx={spineX} cy={y} r={r % 2 === 0 || r === n ? 4 : 2.5} fill={c ?? 'var(--panel-2)'} stroke="url(#resid-grad)" />
            {r >= 0 && r < n && r % 2 === 0 && (
              <text x={spineX - 10} y={y + 4} fontSize={10} textAnchor="end" fill="currentColor" opacity={0.7}>
                {r * 5}%
              </text>
            )}
          </g>
        )
      })}
      <text x={(x0 + x1) / 2} y={yOf(n) - 18} fontSize={11} textAnchor="middle" fill="currentColor" opacity={0.85}>
        output · next token · class · reconstruction · next frame
      </text>
      <text x={(x0 + x1) / 2} y={yOf(-1) + 24} fontSize={11} textAnchor="middle" fill="currentColor" opacity={0.85}>
        input · tokens · pixels · board moves · DNA · video frames
      </text>
      <text x={spineX - 10} y={yOf(n) + 4} fontSize={10} textAnchor="end" fill="currentColor" opacity={0.7}>
        out
      </text>
      <text x={spineX - 10} y={yOf(-1) + 4} fontSize={10} textAnchor="end" fill="currentColor" opacity={0.7}>
        in
      </text>
      {placed.map((p) => (
        <PinCard key={p.entry.slug} p={p} x={p.px} y={p.py} active={expanded === p.entry.slug} onExpand={onExpand} onHover={onHover} anchorX={x1 + 10} anchorY={anchorY(p.key)} />
      ))}
    </svg>
  )
}

/** Left-to-right stages (CNN layers or pipeline modules); box height ∝ √width; pins below. */
function StageDiagram({ m, color, focus, onPick, onHover, pins, expanded, onExpand }: DiagramProps) {
  const stages = m.stages!
  const W = 720
  const gap = 8
  const bw = (W - 20 - gap * (stages.length - 1)) / stages.length
  const maxC = Math.max(...stages.map((s) => s.channels))
  const cy = 110
  const pinTop = 236
  const perRow = Math.floor((W - 20) / (PIN_W + 14)) // pins wrap into rows instead of running off the right edge
  const H = pinTop + Math.ceil(pins.length / perRow) * PIN_STEP + 10
  const xOf = (name: string) => 10 + stages.findIndex((s) => s.name === name) * (bw + gap) + bw / 2
  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }} role="img" aria-label={`${m.name} architecture`}>
      <Defs />
      <line className="flow" x1={10} x2={W - 10} y1={cy} y2={cy} stroke="url(#resid-grad)" strokeWidth={3} />
      {stages.map((s, i) => {
        const key = `stage:${s.name}`
        const h = 30 + 140 * Math.sqrt(s.channels / maxC)
        const x = 10 + i * (bw + gap)
        const c = color(key)
        return (
          <g key={s.name} onClick={() => onPick(key)} onMouseEnter={() => onHover(key)} onMouseLeave={() => onHover(null)} style={{ cursor: 'pointer' }} filter={c ? 'url(#glow)' : undefined}>
            <rect x={x} y={cy - h / 2} width={bw} height={h} rx={8} fill={c ?? 'var(--panel-2)'} stroke={focus === key ? '#f59e0b' : 'currentColor'} strokeOpacity={focus === key ? 1 : 0.2} strokeWidth={focus === key ? 2.5 : 1}>
              <title>
                {s.name} ({s.channels} {m.kind === 'cnn' ? 'channels' : 'dims'})
              </title>
            </rect>
            <text x={x + bw / 2} y={cy + 92} fontSize={10} textAnchor="end" fill="currentColor" transform={`rotate(-35 ${x + bw / 2} ${cy + 92})`}>
              {s.name}
            </text>
          </g>
        )
      })}
      <text x={W / 2} y={16} textAnchor="middle" fontSize={12} fill="currentColor" opacity={0.75}>
        input → output (box height ∝ √width of the layer)
      </text>
      {pins.map((p, i) => (
        <PinCard key={p.entry.slug} p={p} x={10 + (i % perRow) * (PIN_W + 14)} y={pinTop + Math.floor(i / perRow) * PIN_STEP} active={expanded === p.entry.slug} onExpand={onExpand} onHover={onHover} anchorX={xOf(p.key.split(':')[1])} anchorY={cy + 40} />
      ))}
    </svg>
  )
}
