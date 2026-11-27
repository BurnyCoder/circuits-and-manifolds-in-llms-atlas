// Generic matrix heatmap drawn on <canvas> (fast for 113×113) with axes in SVG and a hover tooltip.
// Rows = y (top to bottom), columns = x. Optional outlined cells (e.g. circuit heads) and click handler.
import { useEffect, useRef } from 'react'
import { useTooltip } from './common'

export interface HeatmapProps {
  values: number[][]
  color: (v: number) => string
  width?: number
  cell?: number
  xLabels?: string[]
  yLabels?: string[]
  xTitle?: string
  yTitle?: string
  outline?: Set<string> // "r,c" keys
  selected?: string
  tooltip?: (r: number, c: number, v: number) => React.ReactNode
  onCellClick?: (r: number, c: number) => void
  onCellHover?: (r: number, c: number) => void
  maxLabelEvery?: number
}

export default function Heatmap(p: HeatmapProps) {
  const rows = p.values.length
  const cols = p.values[0]?.length ?? 0
  const cell = p.cell ?? Math.max(2, Math.min(28, Math.floor((p.width ?? 420) / Math.max(cols, 1))))
  const left = p.yLabels || p.yTitle ? 44 : 4
  const top = 4
  const bottom = p.xLabels || p.xTitle ? 34 : 4
  const W = left + cols * cell + 4
  const H = top + rows * cell + bottom
  const canvas = useRef<HTMLCanvasElement>(null)
  const { show, hide, node } = useTooltip()

  useEffect(() => {
    const ctx = canvas.current?.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    canvas.current!.width = cols * cell * dpr
    canvas.current!.height = rows * cell * dpr
    ctx.scale(dpr, dpr)
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        ctx.fillStyle = p.color(p.values[r][c])
        ctx.fillRect(c * cell, r * cell, cell + 0.5, cell + 0.5)
      }
  }, [p.values, p.color, cell, rows, cols])

  const every = p.maxLabelEvery ?? Math.ceil(Math.max(rows, cols) / 24)
  const locate = (e: React.MouseEvent) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const scale = rect.width / (cols * cell)
    const c = Math.floor((e.clientX - rect.left) / scale / cell)
    const r = Math.floor((e.clientY - rect.top) / scale / cell)
    return r >= 0 && r < rows && c >= 0 && c < cols ? [r, c] : null
  }

  return (
    <div style={{ position: 'relative', maxWidth: W, width: '100%' }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }}>
        <foreignObject x={left} y={top} width={cols * cell} height={rows * cell}>
          <canvas
            ref={canvas}
            style={{ width: cols * cell, height: rows * cell, display: 'block', cursor: p.onCellClick ? 'pointer' : 'crosshair' }}
            onMouseMove={(e) => {
              const rc = locate(e)
              if (!rc) return hide()
              p.onCellHover?.(rc[0], rc[1])
              show(e, p.tooltip ? p.tooltip(rc[0], rc[1], p.values[rc[0]][rc[1]]) : p.values[rc[0]][rc[1]].toFixed(3))
            }}
            onMouseLeave={hide}
            onClick={(e) => {
              const rc = locate(e)
              if (rc) p.onCellClick?.(rc[0], rc[1])
            }}
          />
        </foreignObject>
        {[...(p.outline ?? [])].map((k) => {
          const [r, c] = k.split(',').map(Number)
          return <rect key={k} x={left + c * cell} y={top + r * cell} width={cell} height={cell} fill="none" stroke="currentColor" strokeWidth={1.5} pointerEvents="none" />
        })}
        {p.selected &&
          (() => {
            const [r, c] = p.selected.split(',').map(Number)
            return <rect x={left + c * cell - 1} y={top + r * cell - 1} width={cell + 2} height={cell + 2} fill="none" stroke="#f59e0b" strokeWidth={2.5} pointerEvents="none" />
          })()}
        {p.yLabels?.map((l, r) =>
          r % every === 0 ? (
            <text key={r} x={left - 4} y={top + r * cell + cell / 2} fontSize={Math.min(11, cell)} textAnchor="end" dominantBaseline="middle" fill="currentColor">
              {l}
            </text>
          ) : null,
        )}
        {p.xLabels?.map((l, c) =>
          c % every === 0 ? (
            <text key={c} x={left + c * cell + cell / 2} y={top + rows * cell + 12} fontSize={Math.min(11, cell)} textAnchor="middle" fill="currentColor">
              {l}
            </text>
          ) : null,
        )}
        {p.xTitle && (
          <text x={left + (cols * cell) / 2} y={H - 4} fontSize={12} textAnchor="middle" fill="currentColor" opacity={0.75}>
            {p.xTitle}
          </text>
        )}
        {p.yTitle && (
          <text x={10} y={top + (rows * cell) / 2} fontSize={12} textAnchor="middle" fill="currentColor" opacity={0.75} transform={`rotate(-90 10 ${top + (rows * cell) / 2})`}>
            {p.yTitle}
          </text>
        )}
      </svg>
      {node}
    </div>
  )
}

/** Small horizontal color legend for a scale over [lo, hi]. */
export function Legend({ color, lo, hi, label, width = 200 }: { color: (v: number) => string; lo: number; hi: number; label: string; width?: number }) {
  const stops = Array.from({ length: 11 }, (_, i) => lo + ((hi - lo) * i) / 10)
  const id = `lg-${label.replace(/\W/g, '')}-${Math.round(lo * 100)}-${Math.round(hi * 100)}`
  return (
    <svg width={width} height={34} style={{ display: 'block' }} aria-label={`legend: ${label}`}>
      <defs>
        <linearGradient id={id}>
          {stops.map((s, i) => (
            <stop key={i} offset={`${i * 10}%`} stopColor={color(s)} />
          ))}
        </linearGradient>
      </defs>
      <rect x={0} y={2} width={width} height={10} fill={`url(#${id})`} rx={3} />
      <text x={0} y={26} fontSize={11} fill="currentColor">
        {lo.toFixed(2)}
      </text>
      <text x={width / 2} y={26} fontSize={11} fill="currentColor" textAnchor="middle" opacity={0.8}>
        {label}
      </text>
      <text x={width} y={26} fontSize={11} fill="currentColor" textAnchor="end">
        {hi.toFixed(2)}
      </text>
    </svg>
  )
}
