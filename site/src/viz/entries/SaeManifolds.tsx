// SAE manifolds: regime map (manifold × k) and, for a chosen manifold and k, the manifold's points colored by which
// SAE atom fires most — capture (few atoms), shattering (many local patches) or dilution.
import * as d3 from 'd3'
import { useState } from 'react'
import type { VizProps } from '../registry'
import LineChart from '../primitives/LineChart'
import Plot3D from '../primitives/Plot3D'
import { HowTo, Kpis } from '../primitives/common'

interface Row {
  manifold: string
  kind: string
  k_embed: number
  support: number
  spread: number
  r2: number
  regime: string
  intr: number[][]
  atom: number[]
  emb3: number[][]
}
interface D {
  ks: number[]
  table: { k: number; rows: Row[]; frac: Record<string, number>; r2_mean: number }[]
  L0: number
  atoms: number
}
const RC: Record<string, string> = { capture: '#16a34a', shattering: '#dc2626', dilution: '#d97706' }

export default function SaeManifolds({ data }: VizProps) {
  const d = data as unknown as D
  const [ki, setKi] = useState(1)
  const [mi, setMi] = useState(0)
  const t = d.table[ki]
  const r = t.rows[mi]
  const atoms = [...new Set(r.atom)].filter((a) => a >= 0)
  const col = (a: number) => (a < 0 ? '#999' : d3.interpolateRainbow((atoms.indexOf(a) % 12) / 12))
  const e = r.emb3
  return (
    <div>
      <HowTo>
        Synthetic test of whether sparse autoencoders can represent curved concepts. Each training example is a sum of {d.L0}{' '}
        random manifolds (circles, spheres, tori, Möbius strips…) hidden in 128 dimensions; a TopK SAE with {d.atoms} atoms learns
        to reconstruct them. Pick k (atoms active per example) and a manifold: dots are the manifold's points colored by their
        most active atom. Few colors = <b style={{ color: RC.capture }}>capture</b>; many small patches ={' '}
        <b style={{ color: RC.shattering }}>shattering</b>; many broad overlapping atoms = <b style={{ color: RC.dilution }}>dilution</b>.
      </HowTo>
      <div className="viz-controls">
        <label>
          k = {t.k}
          <input type="range" min={0} max={d.table.length - 1} value={ki} onChange={(ev) => setKi(+ev.target.value)} />
        </label>
        <select className="sel" value={mi} onChange={(ev) => setMi(+ev.target.value)}>
          {t.rows.map((x, i) => (
            <option key={x.manifold} value={i}>
              {x.manifold} ({x.k_embed}-D embedding)
            </option>
          ))}
        </select>
      </div>
      <div className="two-col">
        <Plot3D traces={[{ x: e.map((p) => p[0]), y: e.map((p) => p[1] ?? 0), z: e.map((p) => p[2] ?? 0), color: r.atom.map(col), size: 4 }]} axes={['dim 1', 'dim 2', 'dim 3']} height={340} />
        <div>
          <Kpis
            items={[
              ['regime', <span style={{ color: RC[r.regime] }}>{r.regime}</span>],
              ['atoms used (support)', r.support],
              ['receptive-field spread', r.spread.toFixed(2)],
              ['restricted R²', r.r2.toFixed(2)],
            ]}
          />
          <LineChart
            series={['capture', 'shattering', 'dilution'].map((g) => ({ name: g, x: d.table.map((x) => x.k), y: d.table.map((x) => x.frac[g]), color: RC[g] }))}
            xLabel="k (active atoms per example)"
            yLabel="share of manifolds"
            yDomain={[0, 1]}
            marker={t.k}
            fmtX={(v) => String(v)}
            height={220}
          />
        </div>
      </div>
    </div>
  )
}
