// "Model Map": GPT-2 small drawn as a 12×12 grid of attention heads on its residual stream, colored by our measured
// head scores or by circuit roles from the literature. Hover a head to see its real attention pattern on a prompt and
// the atlas entries that involve it. Below: every model used in the atlas, linking to its entries.
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { entries } from '../lib/content'
import { useData } from '../lib/data'
import Heatmap from '../viz/primitives/Heatmap'
import { Tabs, cat10, seqScale } from '../viz/primitives/common'

// Head roles reported in the literature (layer.head), each linked to its entry. Sources: Wang et al. 2022 Fig. 2
// (IOI classes), McDougall et al. 2023 (copy suppression L10H7), Hanna et al. 2023 (greater-than heads).
const ROLES: { role: string; heads: string[]; entry: string }[] = [
  { role: 'IOI name mover', heads: ['9.9', '9.6', '10.0'], entry: 'ioi' },
  { role: 'IOI negative name mover / copy suppression', heads: ['10.7', '11.10'], entry: 'copy-suppression' },
  { role: 'IOI S-inhibition', heads: ['7.3', '7.9', '8.6', '8.10'], entry: 'ioi' },
  { role: 'IOI induction', heads: ['5.5', '6.9', '5.8', '5.9'], entry: 'ioi' },
  { role: 'IOI duplicate token', heads: ['0.1', '3.0', '0.10'], entry: 'ioi' },
  { role: 'IOI previous token', heads: ['2.2', '4.11'], entry: 'ioi' },
  { role: 'IOI backup name mover', heads: ['9.0', '9.7', '10.1', '10.2', '10.6', '10.10', '11.2', '11.9'], entry: 'ioi' },
  { role: 'Greater-than year mover', heads: ['5.1', '5.5', '6.9', '7.10', '8.8', '8.11', '9.1'], entry: 'greater-than' },
]

type Mode = 'induction' | 'prev' | 'duplicate' | 'roles'
interface MapData {
  scores: { prev: number[][]; induction: number[][]; duplicate: number[][] }
  patterns: Record<string, { tokens: string[]; pattern: number[][][][] }>
}

export default function ModelMap() {
  const { data } = useData('gpt2-map.json')
  const [mode, setMode] = useState<Mode>('induction')
  const [prompt, setPrompt] = useState<'ioi' | 'repeat'>('repeat')
  const [head, setHead] = useState<[number, number]>([5, 5])
  const d = data as unknown as MapData | undefined
  const roleOf = useMemo(() => {
    const m: Record<string, number> = {}
    ROLES.forEach((r, i) => r.heads.forEach((h) => (m[h] ??= i)))
    return m
  }, [])
  const color = useMemo(() => seqScale(1), [])
  const models = useMemo(() => {
    const m: Record<string, typeof entries> = {}
    for (const e of entries) for (const name of e.models) (m[name] ??= []).push(e)
    return Object.entries(m).sort((a, b) => b[1].length - a[1].length)
  }, [])

  const key = `${head[0]}.${head[1]}`
  const rolesHere = ROLES.filter((r) => r.heads.includes(key))
  const values = d
    ? mode === 'roles'
      ? d.scores.prev.map((row, l) => row.map((_, h) => (roleOf[`${l}.${h}`] ?? -1) + 1))
      : d.scores[mode]
    : []
  const roleColor = (v: number) => (v === 0 ? 'var(--panel-2)' : cat10[(v - 1) % 10])
  const pat = d?.patterns[prompt]

  return (
    <section className="panel" aria-label="Model map">
      <h2 style={{ marginBottom: 4 }}>Where circuits live: an interactive GPT-2 small</h2>
      <p className="viz-note" style={{ marginTop: 0 }}>
        Each square is one of GPT-2 small's 144 attention heads (rows = layers 0–11, columns = heads). Colors are scores we
        measured in this repo or roles reported in papers. Hover or click a head to see what it attends to.
      </p>
      {!d ? (
        <p>Loading GPT-2 head data…</p>
      ) : (
        <>
          <Tabs
            tabs={[
              ['induction', 'Induction score'],
              ['prev', 'Previous-token score'],
              ['duplicate', 'Duplicate-token score'],
              ['roles', 'Circuit roles (papers)'],
            ]}
            value={mode}
            onChange={setMode}
          />
          <div className="two-col">
            <div>
              <Heatmap
                values={values}
                color={mode === 'roles' ? roleColor : color}
                cell={30}
                yLabels={[...Array(12).keys()].map((i) => `L${i}`)}
                xLabels={[...Array(12).keys()].map(String)}
                xTitle="head"
                yTitle="layer"
                selected={`${head[0]},${head[1]}`}
                onCellHover={(r, c) => setHead([r, c])}
                onCellClick={(r, c) => setHead([r, c])}
                tooltip={(r, c) => (
                  <span>
                    head <b>{r}.{c}</b> · induction {d.scores.induction[r][c].toFixed(2)} · prev {d.scores.prev[r][c].toFixed(2)} · dup{' '}
                    {d.scores.duplicate[r][c].toFixed(2)}
                    {ROLES.filter((x) => x.heads.includes(`${r}.${c}`)).map((x) => (
                      <span key={x.role}>
                        <br />• {x.role}
                      </span>
                    ))}
                  </span>
                )}
              />
              {mode === 'roles' && (
                <div className="viz-controls">
                  {ROLES.map((r, i) => (
                    <span key={r.role} className="tag" style={{ background: cat10[i], color: '#fff' }}>
                      {r.role}
                    </span>
                  ))}
                </div>
              )}
            </div>
            <div>
              <div className="viz-controls">
                <b>Head {key}</b> attention on
                <select className="sel" value={prompt} onChange={(e) => setPrompt(e.target.value as 'ioi' | 'repeat')}>
                  <option value="repeat">a repeated sentence</option>
                  <option value="ioi">an IOI sentence</option>
                </select>
              </div>
              {pat && (
                <Heatmap
                  values={pat.pattern[head[0]][head[1]]}
                  color={color}
                  cell={Math.floor(380 / pat.tokens.length)}
                  yLabels={pat.tokens}
                  xLabels={pat.tokens.map((t) => t.trim().slice(0, 4))}
                  xTitle="attends to (source)"
                  yTitle="from (destination)"
                  maxLabelEvery={1}
                  tooltip={(r, c, v) => `“${pat.tokens[r]}” → “${pat.tokens[c]}”: ${v.toFixed(2)}`}
                />
              )}
              <p className="viz-note">
                {rolesHere.length
                  ? rolesHere.map((r) => (
                      <span key={r.role}>
                        Role: <Link to={`/entry/${r.entry}`}>{r.role}</Link>.{' '}
                      </span>
                    ))
                  : 'No role from the papers covered here; see its scores in the tooltip.'}
              </p>
            </div>
          </div>
        </>
      )}
      <details style={{ marginTop: 8 }}>
        <summary>Other models in the atlas ({models.length})</summary>
        <div className="grid" style={{ marginTop: 8 }}>
          {models.map(([m, es]) => (
            <div key={m} className="kpi">
              <b>{m}</b>
              <div className="viz-note">
                {es.map((e, i) => (
                  <span key={e.slug}>
                    {i ? ' · ' : ''}
                    <Link to={`/entry/${e.slug}`}>{e.title}</Link>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </details>
    </section>
  )
}
