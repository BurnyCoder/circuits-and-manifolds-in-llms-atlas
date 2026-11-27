// Multi-series line chart (D3 scales, React-rendered SVG) with optional log axes, hover crosshair and markers.
import * as d3 from 'd3'
import { useMemo, useState } from 'react'

export interface Series {
  name: string
  x: number[]
  y: number[]
  color: string
  dash?: string
  width?: number
}

export interface LineChartProps {
  series: Series[]
  width?: number
  height?: number
  xLog?: boolean
  yLog?: boolean
  xLabel?: string
  yLabel?: string
  yDomain?: [number, number]
  marker?: number // vertical marker at x
  bands?: { x0: number; x1: number; label: string; color: string }[]
  fmtX?: (v: number) => string
}

export default function LineChart({ series, width = 640, height = 300, xLog, yLog, xLabel, yLabel, yDomain, marker, bands, fmtX }: LineChartProps) {
  const m = { l: 56, r: 16, t: 12, b: 42 }
  const [hx, setHx] = useState<number | null>(null)
  const { xs, ys } = useMemo(() => {
    const allX = series.flatMap((s) => s.x).filter((v) => !xLog || v > 0)
    const allY = series.flatMap((s) => s.y).filter((v) => Number.isFinite(v) && (!yLog || v > 0))
    const xd = d3.extent(allX) as [number, number]
    const yd = yDomain ?? (d3.extent(allY) as [number, number])
    const xs = (xLog ? d3.scaleLog() : d3.scaleLinear()).domain(xd).range([m.l, width - m.r])
    const ys = (yLog ? d3.scaleLog() : d3.scaleLinear()).domain(yd).nice().range([height - m.b, m.t])
    return { xs, ys }
  }, [series, width, height, xLog, yLog, yDomain, m.l, m.r, m.t, m.b])
  const line = (s: Series) =>
    d3
      .line<number>()
      .defined((_, i) => Number.isFinite(s.y[i]) && (!yLog || s.y[i] > 0) && (!xLog || s.x[i] > 0))
      .x((_, i) => xs(s.x[i]))
      .y((v) => ys(v))(s.y) ?? ''
  const fx = fmtX ?? ((v: number) => d3.format('~s')(v))
  // Log axes: label only 1·10^k (plus 3·10^k when the domain spans < 3 decades) to avoid overlapping tick labels.
  const logTicks = (s: d3.ScaleContinuousNumeric<number, number>) => {
    const [a, b] = s.domain()
    const decades = Math.log10(b / a)
    return s.ticks(10).filter((t) => {
      const m = t / 10 ** Math.floor(Math.log10(t) + 1e-9)
      return Math.abs(m - 1) < 1e-6 || (decades < 3 && Math.abs(m - 3) < 1e-6)
    })
  }
  const xTicks = xLog ? logTicks(xs) : xs.ticks(6)
  const yTicks = yLog ? logTicks(ys) : ys.ticks(5)
  const nearest = (s: Series, x: number) => {
    let best = 0
    s.x.forEach((v, i) => (Math.abs(v - x) < Math.abs(s.x[best] - x) ? (best = i) : 0))
    return best
  }
  return (
    <div style={{ width: '100%', maxWidth: width }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        style={{ width: '100%' }}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect()
          const px = ((e.clientX - r.left) / r.width) * width
          if (px >= m.l && px <= width - m.r) setHx(xs.invert(px))
        }}
        onMouseLeave={() => setHx(null)}
      >
        {bands?.map((b) => (
          <g key={b.label}>
            <rect x={xs(b.x0)} y={m.t} width={Math.max(1, xs(b.x1) - xs(b.x0))} height={height - m.t - m.b} fill={b.color} opacity={0.15} />
            <text x={xs(b.x0) + 4} y={m.t + 12} fontSize={11} fill="currentColor" opacity={0.8}>
              {b.label}
            </text>
          </g>
        ))}
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={width - m.r} y1={ys(t)} y2={ys(t)} stroke="currentColor" opacity={0.1} />
            <text x={m.l - 6} y={ys(t)} fontSize={11} textAnchor="end" dominantBaseline="middle" fill="currentColor">
              {yLog ? d3.format('.0e')(t) : d3.format('~g')(t)}
            </text>
          </g>
        ))}
        {xTicks.map((t) => (
          <text key={t} x={xs(t)} y={height - m.b + 16} fontSize={11} textAnchor="middle" fill="currentColor">
            {fx(t)}
          </text>
        ))}
        <line x1={m.l} x2={width - m.r} y1={height - m.b} y2={height - m.b} stroke="currentColor" opacity={0.4} />
        <line x1={m.l} x2={m.l} y1={m.t} y2={height - m.b} stroke="currentColor" opacity={0.4} />
        {xLabel && (
          <text x={(m.l + width - m.r) / 2} y={height - 6} fontSize={12} textAnchor="middle" fill="currentColor" opacity={0.75}>
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text x={14} y={(m.t + height - m.b) / 2} fontSize={12} textAnchor="middle" fill="currentColor" opacity={0.75} transform={`rotate(-90 14 ${(m.t + height - m.b) / 2})`}>
            {yLabel}
          </text>
        )}
        {series.map((s) => (
          <path key={s.name} d={line(s)} fill="none" stroke={s.color} strokeWidth={s.width ?? 2} strokeDasharray={s.dash} />
        ))}
        {marker !== undefined && <line x1={xs(marker)} x2={xs(marker)} y1={m.t} y2={height - m.b} stroke="#f59e0b" strokeWidth={2} />}
        {hx !== null && (
          <g pointerEvents="none">
            <line x1={xs(hx)} x2={xs(hx)} y1={m.t} y2={height - m.b} stroke="currentColor" opacity={0.3} />
            {series.map((s, k) => {
              const i = nearest(s, hx)
              return (
                <g key={s.name}>
                  <circle cx={xs(s.x[i])} cy={ys(s.y[i])} r={3.5} fill={s.color} />
                  <text x={Math.min(xs(hx) + 8, width - 150)} y={m.t + 14 + k * 14} fontSize={11} fill={s.color}>
                    {s.name}: {d3.format('.3~g')(s.y[i])} @ {fx(s.x[i])}
                  </text>
                </g>
              )
            })}
          </g>
        )}
      </svg>
      <div className="viz-controls" style={{ marginTop: 0 }}>
        {series.map((s) => (
          <span key={s.name} style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
            <svg width={22} height={8}>
              <line x1={0} x2={22} y1={4} y2={4} stroke={s.color} strokeWidth={3} strokeDasharray={s.dash} />
            </svg>
            {s.name}
          </span>
        ))}
      </div>
    </div>
  )
}
