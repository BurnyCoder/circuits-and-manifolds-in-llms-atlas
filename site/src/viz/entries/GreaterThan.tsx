// Greater-than entry: year × next-year probability heatmap; toggle patching of the circuit heads and MLPs 8–11 with
// activations from the corrupted "1701" prompt and watch the sharp edge at the start year disappear.
import * as d3 from 'd3'
import { useMemo, useState } from 'react'
import type { VizProps } from '../registry'
import Heatmap, { Legend } from '../primitives/Heatmap'
import { HowTo, Kpis, seqScale } from '../primitives/common'

interface D {
  years: number[]
  groups: string[]
  heads: string[]
  combos: { mask: number[]; pd: number; matrix: number[][] }[]
  rand_pd: number[]
  summary: Record<string, number>
}

export default function GreaterThan({ data }: VizProps) {
  const d = data as unknown as D
  const [mask, setMask] = useState<number[]>(d.groups.map(() => 0))
  const combo = d.combos.find((c) => c.mask.every((v, i) => v === mask[i]))!
  const vmax = useMemo(() => Math.max(...d.combos[0].matrix.flat()), [d])
  const color = useMemo(() => seqScale(vmax, d3.interpolateMagma), [vmax])
  return (
    <div>
      <HowTo>
        Rows: the start year in “The war lasted from the year 17<b>YY</b> to the year 17…”. Columns: GPT-2 small's
        probability for each two-digit continuation 00–99. A model that “knows” greater-than puts its mass to the{' '}
        <b>right of the diagonal</b>. Tick components to replace them with their activations on the uninformative prompt
        “…from the year 1701…” (activation patching) and see which ones carry the comparison.
      </HowTo>
      <div className="viz-controls">
        {d.groups.map((g, i) => (
          <label key={g}>
            <input type="checkbox" checked={!!mask[i]} onChange={() => setMask((m) => m.map((v, j) => (j === i ? 1 - v : v)))} />
            patch {g}
            {i === 0 ? ` (${d.heads.join(', ')})` : ''}
          </label>
        ))}
      </div>
      <Heatmap
        values={combo.matrix}
        color={color}
        cell={5}
        yLabels={d.years.map((y) => `17${String(y).padStart(2, '0')}`)}
        xLabels={[...Array(100).keys()].map((y) => String(y).padStart(2, '0'))}
        xTitle="predicted next two digits"
        yTitle="start year"
        maxLabelEvery={10}
        tooltip={(r, c, v) => `start 17${String(d.years[r]).padStart(2, '0')} → “${String(c).padStart(2, '0')}”: p = ${v.toFixed(3)}`}
      />
      <Legend color={color} lo={0} hi={vmax} label="probability (renormalized over 00–99)" />
      <Kpis
        items={[
          ['probability difference P(>YY) − P(≤YY), clean', `${(d.summary.pd_clean * 100).toFixed(1)}%`],
          ['with your patches', `${(combo.pd * 100).toFixed(1)}%`],
          [`${d.heads.length} random heads patched (median of ${d.rand_pd.length})`, `${((d3.median(d.rand_pd) ?? NaN) * 100).toFixed(1)}%`],
          ['circuit only (everything else patched)', `${(d.summary.pd_circuit_only * 100).toFixed(1)}%`],
        ]}
      />
    </div>
  )
}
