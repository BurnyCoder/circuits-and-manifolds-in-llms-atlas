// Curve detectors entry: radial (polar) tuning curves for each InceptionV1 mixed3b curve neuron, the joint tuning
// of all ten (tiling the circle), control neurons, a live stimulus preview, and our Lucent feature visualizations.
import * as d3 from 'd3'
import { useState } from 'react'
import type { VizProps } from '../registry'
import { HowTo, Kpis, Tabs } from '../primitives/common'

interface N {
  neuron: number
  is_curve: boolean
  best_radius: number
  pref_orient: number
  selectivity: number
  curve_over_line: number
  tuning: number[][] // [radius][orient]
  line: number[]
}
interface D {
  radii: number[]
  orients: number[]
  neurons: N[]
  feature_vis: Record<string, string>
  summary: Record<string, number>
}

const hue = (deg: number) => d3.interpolateRainbow(deg / 360)

function Polar({ series, size = 320 }: { series: { r: number[]; color: string; width?: number; dash?: string }[]; size?: number }) {
  const max = Math.max(...series.flatMap((s) => s.r), 1e-6)
  const c = size / 2
  const R = c - 16
  const pts = (r: number[]) => r.map((v, i) => [c + (Math.max(v, 0) / max) * R * Math.cos((i * 5 * Math.PI) / 180), c - (Math.max(v, 0) / max) * R * Math.sin((i * 5 * Math.PI) / 180)])
  return (
    <svg viewBox={`0 0 ${size} ${size}`} style={{ width: '100%', maxWidth: size }}>
      {[0.25, 0.5, 0.75, 1].map((f) => (
        <circle key={f} cx={c} cy={c} r={R * f} fill="none" stroke="currentColor" opacity={0.12} />
      ))}
      {[0, 90, 180, 270].map((a) => (
        <text key={a} x={c + (R + 8) * Math.cos((a * Math.PI) / 180)} y={c - (R + 8) * Math.sin((a * Math.PI) / 180)} fontSize={10} textAnchor="middle" dominantBaseline="middle" fill="currentColor" opacity={0.7}>
          {a}°
        </text>
      ))}
      {series.map((s, k) => (
        <path key={k} d={(d3.line()(pts(s.r) as [number, number][]) ?? '') + 'Z'} fill={s.color} fillOpacity={series.length === 1 ? 0.25 : 0.08} stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash} />
      ))}
    </svg>
  )
}

function Stimulus({ theta, radius }: { theta: number; radius: number | null }) {
  // Same geometry as the Python stimuli: arc midpoint at the center, bulging toward θ.
  const S = 224
  const t = (theta * Math.PI) / 180
  let path: string
  if (radius === null) {
    const dx = -Math.sin(t) * 60
    const dy = -Math.cos(t) * 60
    path = `M${112 - dx},${112 - dy} L${112 + dx},${112 + dy}`
  } else {
    const ox = 112 - radius * Math.cos(t)
    const oy = 112 + radius * Math.sin(t)
    const span = Math.min(70, (60 / radius) * (180 / Math.PI)) * (Math.PI / 180)
    const a0 = -t - span
    const a1 = -t + span
    path = `M${ox + radius * Math.cos(a0)},${oy + radius * Math.sin(a0)} A${radius},${radius} 0 0 1 ${ox + radius * Math.cos(a1)},${oy + radius * Math.sin(a1)}`
  }
  return (
    <svg viewBox={`0 0 ${S} ${S}`} style={{ width: 120, background: '#fff', borderRadius: 8 }}>
      <path d={path} stroke="#000" strokeWidth={4} fill="none" />
    </svg>
  )
}

export default function CurveDetectors({ data }: VizProps) {
  const d = data as unknown as D
  const curves = d.neurons.filter((n) => n.is_curve)
  const controls = d.neurons.filter((n) => !n.is_curve)
  const [tab, setTab] = useState<'one' | 'all' | 'vis'>('all')
  const [sel, setSel] = useState(curves[0].neuron)
  const [ri, setRi] = useState(1)
  const [theta, setTheta] = useState(90)
  const n = d.neurons.find((x) => x.neuron === sel)!
  const oi = Math.round(theta / 5) % d.orients.length
  return (
    <div>
      <Tabs
        tabs={[
          ['all', '1 · Ten neurons tile the circle'],
          ['one', '2 · One neuron, up close'],
          ['vis', '3 · What each neuron wants to see'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'all' && (
        <>
          <HowTo>
            We show InceptionV1 synthetic curves (arcs) at every orientation and read each curve neuron in layer mixed3b.
            Each colored petal is one neuron's response by orientation (at its best radius), normalized to its peak and
            colored by its preferred orientation. Together the ten petals cover the full 360°. Dashed gray: 20 random
            other mixed3b neurons (control) — mostly round, i.e. not orientation-tuned.
          </HowTo>
          <div className="two-col">
            <Polar
              series={curves.map((c) => {
                const r = c.tuning[d.radii.indexOf(c.best_radius)]
                const m = Math.max(...r)
                return { r: r.map((v) => v / m), color: hue(c.pref_orient) }
              })}
            />
            <Polar
              series={controls.slice(0, 10).map((c) => {
                const r = c.tuning[d.radii.indexOf(c.best_radius)]
                const m = Math.max(...r, 1e-6)
                return { r: r.map((v) => v / m), color: '#94a3b8', width: 1, dash: '3 3' }
              })}
            />
          </div>
          <Kpis
            items={[
              ['median orientation selectivity — curve neurons', d.summary.sel_curve_median.toFixed(2)],
              ['… random control neurons', d.summary.sel_control_median.toFixed(2)],
              ['largest gap between preferred orientations', `${d.summary.max_orient_gap.toFixed(0)}°`],
              ['neurons preferring curves > 1.5× over lines', `${d.summary.n_curve_gt_line}/10`],
            ]}
          />
        </>
      )}
      {tab === 'one' && (
        <>
          <HowTo>
            Pick a neuron and a curve radius. The polar plot shows its activation for every orientation; the dashed line is
            its response to straight lines. Drag the orientation slider to preview the exact stimulus.
          </HowTo>
          <div className="viz-controls">
            <select className="sel" value={sel} onChange={(e) => setSel(+e.target.value)}>
              {d.neurons.map((x) => (
                <option key={x.neuron} value={x.neuron}>
                  mixed3b:{x.neuron} {x.is_curve ? '(curve detector)' : '(control)'}
                </option>
              ))}
            </select>
            <label>
              radius
              <select className="sel" value={ri} onChange={(e) => setRi(+e.target.value)}>
                {d.radii.map((r, i) => (
                  <option key={r} value={i}>
                    {r}px
                  </option>
                ))}
              </select>
            </label>
            <label>
              stimulus orientation {theta}°
              <input type="range" min={0} max={355} step={5} value={theta} onChange={(e) => setTheta(+e.target.value)} />
            </label>
          </div>
          <div className="two-col">
            <Polar
              series={[
                { r: n.tuning[ri], color: n.is_curve ? hue(n.pref_orient) : '#94a3b8', width: 2.5 },
                { r: n.line, color: 'currentColor', width: 1.5, dash: '5 4' },
              ]}
            />
            <div>
              <Stimulus theta={theta} radius={d.radii[ri]} />
              <Kpis
                items={[
                  ['activation for this stimulus', n.tuning[ri][oi].toFixed(1)],
                  ['preferred orientation', `${n.pref_orient}°`],
                  ['selectivity (0 = none, 1 = one angle)', n.selectivity.toFixed(2)],
                  ['best curve ÷ best straight line', n.curve_over_line.toFixed(1)],
                ]}
              />
            </div>
          </div>
        </>
      )}
      {tab === 'vis' && (
        <>
          <HowTo>
            Feature visualizations we rendered with Lucent: an image optimized to maximally activate each neuron. Every curve
            detector draws curves of its preferred orientation, often with small perpendicular lines (“combing”).
          </HowTo>
          <div className="grid" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(min(100%, 150px), 1fr))' }}>
            {curves.map((c) => (
              <figure key={c.neuron} style={{ margin: 0, textAlign: 'center' }}>
                <img src={`${import.meta.env.BASE_URL}${d.feature_vis[String(c.neuron)]}`} alt={`feature visualization of mixed3b:${c.neuron}`} style={{ width: '100%', borderRadius: 10, border: `3px solid ${hue(c.pref_orient)}` }} />
                <figcaption className="viz-note">
                  mixed3b:{c.neuron} · prefers {c.pref_orient}°
                </figcaption>
              </figure>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
