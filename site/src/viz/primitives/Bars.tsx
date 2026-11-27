// Horizontal bar chart with optional 95% CI whiskers (value, lo, hi). Used for ablations and replication numbers.
import * as d3 from 'd3'

export interface Bar {
  label: string
  value: number
  lo?: number
  hi?: number
  color?: string
}

export default function Bars({ bars, width = 560, xLabel, domain }: { bars: Bar[]; width?: number; xLabel?: string; domain?: [number, number] }) {
  const rowH = 30
  const m = { l: 210, r: 60, t: 6, b: 34 }
  const H = m.t + bars.length * rowH + m.b
  const hi = domain?.[1] ?? Math.max(...bars.map((b) => b.hi ?? b.value), 0)
  const lo = domain?.[0] ?? Math.min(0, ...bars.map((b) => b.lo ?? b.value))
  const x = d3.scaleLinear().domain([lo, hi]).nice().range([m.l, width - m.r])
  return (
    <svg viewBox={`0 0 ${width} ${H}`} style={{ width: '100%', maxWidth: width }}>
      {x.ticks(5).map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={m.t} y2={H - m.b} stroke="currentColor" opacity={0.1} />
          <text x={x(t)} y={H - m.b + 14} fontSize={11} textAnchor="middle" fill="currentColor">
            {d3.format('~g')(t)}
          </text>
        </g>
      ))}
      {bars.map((b, i) => {
        const y = m.t + i * rowH
        return (
          <g key={b.label}>
            <text x={m.l - 8} y={y + rowH / 2} fontSize={12} textAnchor="end" dominantBaseline="middle" fill="currentColor">
              {b.label}
            </text>
            <rect x={Math.min(x(0), x(b.value))} y={y + 5} width={Math.abs(x(b.value) - x(0))} height={rowH - 10} rx={4} fill={b.color ?? 'var(--accent)'} opacity={0.85} />
            {b.lo !== undefined && b.hi !== undefined && (
              <g stroke="currentColor" strokeWidth={1.5}>
                <line x1={x(b.lo)} x2={x(b.hi)} y1={y + rowH / 2} y2={y + rowH / 2} />
                <line x1={x(b.lo)} x2={x(b.lo)} y1={y + 9} y2={y + rowH - 9} />
                <line x1={x(b.hi)} x2={x(b.hi)} y1={y + 9} y2={y + rowH - 9} />
              </g>
            )}
            <text x={Math.max(x(b.value), x(b.hi ?? b.value)) + 6} y={y + rowH / 2} fontSize={12} dominantBaseline="middle" fill="currentColor">
              {d3.format('.3~g')(b.value)}
            </text>
          </g>
        )
      })}
      {xLabel && (
        <text x={(m.l + width - m.r) / 2} y={H - 4} fontSize={12} textAnchor="middle" fill="currentColor" opacity={0.75}>
          {xLabel}
        </text>
      )}
    </svg>
  )
}
