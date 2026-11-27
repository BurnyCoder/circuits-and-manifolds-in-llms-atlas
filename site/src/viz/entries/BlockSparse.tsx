// Block-sparse featurizers: per-concept R² of a block-sparse featurizer vs a TopK SAE on multi-dimensional toy
// concepts, and InceptionV1's curve responses forming an orientation ring described by a few harmonics.
import * as d3 from 'd3'
import type { VizProps } from '../registry'
import Bars from '../primitives/Bars'
import Scatter2D from '../primitives/Scatter2D'
import { HowTo, Kpis } from '../primitives/common'

interface D {
  toy: Record<string, number[]>
  curve: { harmonics: number[]; total: number; null_mean: number; null_p95: number; ring: [number, number][]; orient_deg: number[] }
}

export default function BlockSparse({ data }: VizProps) {
  const d = data as unknown as D
  const names = Object.keys(d.toy)
  const med = (a: number[]) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)]
  return (
    <div>
      <HowTo>
        Left: toy concepts that each live on a sphere in their own 3-D subspace. A block-sparse featurizer (latents grouped into
        blocks of 4, whole blocks switched on or off) keeps each concept in one block; a standard TopK SAE with the same budget
        splits it across single directions. Right: InceptionV1's responses to curves at every orientation, projected on two
        principal components — a ring colored by orientation.
      </HowTo>
      <div className="two-col">
        <div>
          <Bars bars={names.map((n, i) => ({ label: n, value: med(d.toy[n]), color: i === 0 ? 'var(--c-feature)' : '#94a3b8' }))} xLabel="median per-concept R² (toy)" domain={[0, 1]} />
          <Bars bars={d.curve.harmonics.map((h, i) => ({ label: `orientation harmonic ${i + 1}`, value: h, color: 'var(--c-manifold)' }))} xLabel="variance of the 16-D curve block explained" domain={[0, 1]} />
          <Kpis
            items={[
              ['harmonics 1–3 total', `${(d.curve.total * 100).toFixed(0)}%`],
              ['shuffled-orientation null (95th pct)', `${(d.curve.null_p95 * 100).toFixed(0)}%`],
            ]}
          />
        </div>
        <Scatter2D xy={d.curve.ring} colors={d.curve.orient_deg.map((o) => d3.interpolateRainbow(o / 360))} size={4} />
      </div>
    </div>
  )
}
