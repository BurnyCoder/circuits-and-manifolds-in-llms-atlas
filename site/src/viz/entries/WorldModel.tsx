// World model: frame embeddings in 3-D colored by the car's position (a 1-D curve), decoded frames along the manifold
// vs the straight line between two positions, isometry and probe numbers.
import * as d3 from 'd3'
import type { VizProps } from '../registry'
import Plot3D from '../primitives/Plot3D'
import { HowTo, Kpis } from '../primitives/common'

interface D {
  pcs: number[][]
  pos: number[]
  manifold_img: string
  linear_img: string
  path_positions: number[]
  probes: Record<string, number>
  summary: Record<string, number>
}

export default function WorldModel({ data }: VizProps) {
  const d = data as unknown as D
  const lo = Math.min(...d.pos)
  const hi = Math.max(...d.pos)
  return (
    <div>
      <HowTo>
        A small network learned to predict the next video frame of the classic “mountain car” game. Each dot is one frame's
        internal embedding, colored by where the car is on the hill: the frames line up along a single curved string. Below,
        frames decoded while walking from the valley to the slope <b>along that curve</b> vs <b>straight through embedding
        space</b>.
      </HowTo>
      <Plot3D traces={[{ x: d.pcs.map((p) => p[0]), y: d.pcs.map((p) => p[1]), z: d.pcs.map((p) => p[2]), color: d.pos.map((p) => d3.interpolateViridis((p - lo) / (hi - lo))), size: 2.5 }]} height={380} />
      <h3>Decoded frames along the manifold</h3>
      <img src={`${import.meta.env.BASE_URL}${d.manifold_img}`} alt="frames decoded along the manifold path" style={{ width: '100%', imageRendering: 'pixelated', borderRadius: 8 }} />
      <h3>Decoded frames along the straight line</h3>
      <img src={`${import.meta.env.BASE_URL}${d.linear_img}`} alt="frames decoded along the straight-line path" style={{ width: '100%', imageRendering: 'pixelated', borderRadius: 8 }} />
      <Kpis
        items={[
          ['position probe R²', d.summary.r2_position.toFixed(2)],
          ['velocity probe R²', d.summary.r2_velocity.toFixed(2)],
          ['distance-in-world correlation: along curve', d.summary.r_arc.toFixed(3)],
          ['… straight chord', d.summary.r_chord.toFixed(3)],
          ['mean distance to real frames: curve / line', `${d.summary.void_manifold.toFixed(2)} / ${d.summary.void_linear.toFixed(2)}`],
        ]}
      />
    </div>
  )
}
