// IOI entry: (1) clickable circuit diagram — knock out head classes and read the real logit difference from the
// precomputed 2^7 combinations, (2) per-head importance maps (direct logit attribution, activation noising),
// (3) name-mover attention on an example sentence.
import { useMemo, useState } from 'react'
import type { VizProps } from '../registry'
import Heatmap, { Legend } from '../primitives/Heatmap'
import TokenAttention from '../primitives/TokenAttention'
import { HowTo, Kpis, Tabs, cat10, divScale } from '../primitives/common'

interface D {
  classes: Record<string, string[]>
  dla: number[][]
  noise: number[][]
  combos: { mask: number[]; ld: number }[]
  rand_ablated: number[]
  nm_attn: Record<string, { IO: number; S1: number; S2: number }>
  example: { text: string; tokens: string[]; io_pos: number; s_pos: number[]; end_attn: Record<string, number[]> }
  summary: Record<string, number>
}

// Node layout (x = rough layer position, y = lane), edges as in Wang et al. 2022 Fig. 2.
const LAYOUT: Record<string, [number, number]> = {
  'Duplicate token': [0.08, 0.2],
  'Previous token': [0.2, 0.75],
  Induction: [0.38, 0.75],
  'S-inhibition': [0.56, 0.45],
  'Name mover': [0.76, 0.2],
  'Negative name mover': [0.76, 0.5],
  'Backup name mover': [0.76, 0.82],
}
const EDGES: [string, string][] = [
  ['Previous token', 'Induction'],
  ['Induction', 'S-inhibition'],
  ['Duplicate token', 'S-inhibition'],
  ['S-inhibition', 'Name mover'],
  ['S-inhibition', 'Negative name mover'],
  ['S-inhibition', 'Backup name mover'],
]

export default function IOI({ data }: VizProps) {
  const d = data as unknown as D
  const names = Object.keys(d.classes)
  const [off, setOff] = useState<Set<string>>(new Set())
  const [tab, setTab] = useState<'circuit' | 'heads' | 'attn'>('circuit')
  const [metric, setMetric] = useState<'dla' | 'noise'>('dla')
  const nmHeads = Object.keys(d.example.end_attn)
  const [nm, setNm] = useState(nmHeads[0])

  const full = d.combos[0].ld
  const mask = names.map((n) => (off.has(n) ? 1 : 0))
  const cur = d.combos.find((c) => c.mask.every((v, i) => v === mask[i]))?.ld ?? full
  const sortedRand = [...d.rand_ablated].sort((a, b) => a - b)
  const randMed = sortedRand[Math.floor(sortedRand.length / 2)]
  const classOf = useMemo(() => {
    const m: Record<string, string> = {}
    for (const [k, hs] of Object.entries(d.classes)) for (const h of hs) m[h] = k
    return m
  }, [d.classes])
  const outline = useMemo(() => new Set(Object.keys(classOf).map((h) => h.split('.').join(','))), [classOf])
  const mat = metric === 'dla' ? d.dla : d.noise
  const absMax = Math.max(...mat.flat().map(Math.abs))
  const color = useMemo(() => divScale(absMax), [absMax])
  const W = 720
  const H = 320

  return (
    <div>
      <Tabs
        tabs={[
          ['circuit', '1 · Knock out the circuit'],
          ['heads', '2 · Which heads matter'],
          ['attn', '3 · Name movers look at the answer'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'circuit' && (
        <>
          <HowTo>
            Click a class of heads to <b>mean-ablate</b> it (replace its output with its average on prompts with three
            different names). The number is GPT-2 small's real average logit difference logit(IO) − logit(S), recomputed for
            every one of the 128 on/off combinations. Higher = more confident in the correct name.
          </HowTo>
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: W }}>
            <defs>
              <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill="currentColor" opacity={0.6} />
              </marker>
            </defs>
            {EDGES.map(([a, b]) => (
              <line
                key={a + b}
                x1={LAYOUT[a][0] * W + 70}
                y1={LAYOUT[a][1] * H}
                x2={LAYOUT[b][0] * W - 72}
                y2={LAYOUT[b][1] * H}
                stroke="currentColor"
                opacity={off.has(a) || off.has(b) ? 0.12 : 0.5}
                strokeWidth={2}
                markerEnd="url(#arr)"
              />
            ))}
            {['Name mover', 'Negative name mover', 'Backup name mover'].map((n) => (
              <line key={n} x1={LAYOUT[n][0] * W + 72} y1={LAYOUT[n][1] * H} x2={W - 60} y2={H / 2} stroke="currentColor" opacity={off.has(n) ? 0.12 : 0.5} strokeWidth={2} markerEnd="url(#arr)" />
            ))}
            <g transform={`translate(${W - 56}, ${H / 2})`}>
              <rect x={-4} y={-22} width={58} height={44} rx={10} fill="var(--panel-2)" stroke="currentColor" opacity={0.9} />
              <text x={25} y={-4} fontSize={12} textAnchor="middle" fill="currentColor">
                logits
              </text>
              <text x={25} y={12} fontSize={11} textAnchor="middle" fill="currentColor" opacity={0.7}>
                “ Mary”
              </text>
            </g>
            {names.map((n, i) => {
              const [x, y] = LAYOUT[n]
              const isOff = off.has(n)
              return (
                <g
                  key={n}
                  transform={`translate(${x * W}, ${y * H})`}
                  style={{ cursor: 'pointer' }}
                  onClick={() =>
                    setOff((s) => {
                      const t = new Set(s)
                      if (t.has(n)) t.delete(n)
                      else t.add(n)
                      return t
                    })
                  }
                >
                  <rect x={-72} y={-26} width={144} height={52} rx={12} fill={isOff ? 'var(--panel-2)' : cat10[i]} opacity={isOff ? 1 : 0.9} stroke={isOff ? 'currentColor' : 'none'} strokeDasharray={isOff ? '4 3' : undefined} />
                  <text y={-6} textAnchor="middle" fontSize={12.5} fontWeight={700} fill={isOff ? 'currentColor' : '#fff'}>
                    {n}
                  </text>
                  <text y={12} textAnchor="middle" fontSize={11} fill={isOff ? 'currentColor' : '#fff'} opacity={0.9}>
                    {isOff ? 'ablated' : d.classes[n].join(' ')}
                  </text>
                </g>
              )
            })}
          </svg>
          <Kpis
            items={[
              ['full model', full.toFixed(2)],
              ['with your knockouts', <span style={{ color: cur < 0.5 * full ? 'var(--bad)' : undefined }}>{cur.toFixed(2)}</span>],
              [`${d.rand_ablated.length} random sets of 26 heads (median)`, randMed.toFixed(2)],
              ['every class knocked out', d.combos[d.combos.length - 1].ld.toFixed(2)],
            ]}
          />
          <div className="viz-controls">
            <button className="chip" onClick={() => setOff(new Set())}>
              reset
            </button>
            <button className="chip" onClick={() => setOff(new Set(names))}>
              ablate all
            </button>
          </div>
        </>
      )}
      {tab === 'heads' && (
        <>
          <HowTo>
            Rows = layers, columns = heads; outlined cells are the 26 circuit heads. <b>Direct logit attribution</b>: how much
            each head's output at the last token directly pushes logit(IO) − logit(S). <b>Noising</b>: change in logit
            difference when that head's output is replaced by its output on a prompt with three unrelated names (total
            effect, including through later heads). Red = helps the correct answer, blue = hurts it.
          </HowTo>
          <div className="viz-controls">
            <label>
              <input type="radio" checked={metric === 'dla'} onChange={() => setMetric('dla')} /> direct logit attribution
            </label>
            <label>
              <input type="radio" checked={metric === 'noise'} onChange={() => setMetric('noise')} /> activation noising (Δ logit diff)
            </label>
          </div>
          <Heatmap
            values={mat}
            color={color}
            cell={30}
            outline={outline}
            yLabels={[...Array(12).keys()].map((i) => `L${i}`)}
            xLabels={[...Array(12).keys()].map(String)}
            xTitle="head"
            yTitle="layer"
            tooltip={(r, c, v) => (
              <span>
                head <b>{r}.{c}</b>: {v.toFixed(3)}
                <br />
                {classOf[`${r}.${c}`] ?? 'not in the circuit'}
              </span>
            )}
          />
          <Legend color={color} lo={-absMax} hi={absMax} label={metric === 'dla' ? 'logit units' : 'Δ logit diff'} />
        </>
      )}
      {tab === 'attn' && (
        <>
          <HowTo>
            Attention from the final token (“ to”) for a name-mover head. Green outline = indirect object (the right
            answer), red outline = subject. Name movers attend to the IO and copy it; negative name movers attend to it too
            but write <i>against</i> it (hedging).
          </HowTo>
          <div className="viz-controls">
            {nmHeads.map((h) => (
              <label key={h}>
                <input type="radio" checked={nm === h} onChange={() => setNm(h)} /> {h} ({classOf[h]?.toLowerCase()})
              </label>
            ))}
          </div>
          <TokenAttention
            tokens={d.example.tokens}
            pattern={d.example.tokens.map((_, i) => (i === d.example.tokens.length - 1 ? d.example.end_attn[nm] : d.example.tokens.map(() => 0)))}
            initial={d.example.tokens.length - 1}
            highlight={{ [d.example.io_pos]: '#16a34a', ...Object.fromEntries(d.example.s_pos.map((p) => [p, '#dc2626'])) }}
          />
          <Kpis
            items={[
              [`${nm} → IO (mean over prompts)`, d.nm_attn[nm].IO.toFixed(2)],
              [`${nm} → S (1st)`, d.nm_attn[nm].S1.toFixed(2)],
              [`${nm} → S (2nd)`, d.nm_attn[nm].S2.toFixed(2)],
            ]}
          />
        </>
      )}
    </div>
  )
}
