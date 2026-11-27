// Colour bicone: FLUX VAE latents of solid colors in their top-3 principal components, drawn in their true colors.
import { useState } from 'react'
import type { VizProps } from '../registry'
import Plot3D from '../primitives/Plot3D'
import Scatter2D from '../primitives/Scatter2D'
import { HowTo, Kpis } from '../primitives/common'

interface D {
  pcs: [number, number, number][]
  rgb: [number, number, number][]
  hls: [number, number, number][]
  var: number[]
  var_noise: number[]
  summary: Record<string, number>
}

export default function ColourBicone({ data }: VizProps) {
  const d = data as unknown as D
  const [view, setView] = useState<'3d' | 'wheel'>('3d')
  const colors = d.rgb.map(([r, g, b]) => `rgb(${r},${g},${b})`)
  const text = d.hls.map(([h, l, s]) => `hue ${(h * 360).toFixed(0)}°, lightness ${(l * 100).toFixed(0)}%, saturation ${(s * 100).toFixed(0)}%`)
  return (
    <div>
      <HowTo>
        {d.summary.n_colors} solid-color images, encoded by the FLUX image autoencoder and averaged over space. Each dot is one
        image, drawn in its own color, at its position along the top three principal components of the latent. Black and
        white sit at the two tips, gray in the middle, and hues go around: an <b>HSL bicone</b>. Drag to rotate.
      </HowTo>
      <div className="viz-controls">
        <label>
          <input type="radio" checked={view === '3d'} onChange={() => setView('3d')} /> 3-D bicone
        </label>
        <label>
          <input type="radio" checked={view === 'wheel'} onChange={() => setView('wheel')} /> top view (PC2 × PC3): the hue wheel
        </label>
      </div>
      {view === '3d' ? (
        <Plot3D traces={[{ x: d.pcs.map((p) => p[1]), y: d.pcs.map((p) => p[2]), z: d.pcs.map((p) => p[0]), color: colors, size: 5, text }]} axes={['PC2', 'PC3', 'PC1 (lightness)']} height={460} />
      ) : (
        <Scatter2D xy={d.pcs.map((p) => [p[1], p[2]] as [number, number])} colors={colors} size={5} alpha={0.95} />
      )}
      <Kpis
        items={[
          ['variance in top 3 PCs', `${(d.summary.var3 * 100).toFixed(2)}%`],
          ['… for random-noise images (control)', `${(d.summary.var_noise3 * 100).toFixed(1)}%`],
          ['corr(PC1, lightness)', d.summary.r_light.toFixed(2)],
          ['circular corr(angle, hue)', d.summary.r_hue.toFixed(2)],
          ['corr(radius, chroma)', d.summary.r_sat.toFixed(2)],
        ]}
      />
    </div>
  )
}
