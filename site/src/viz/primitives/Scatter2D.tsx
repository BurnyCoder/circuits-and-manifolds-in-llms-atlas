// Fast 2-D scatter on <canvas> (thousands of points) with optional class-mean labels and connecting polygon.
import { useEffect, useRef } from 'react'

export interface Scatter2DProps {
  xy: [number, number][]
  colors: string[]
  size?: number
  width?: number
  height?: number
  labels?: { xy: [number, number]; text: string; color?: string }[]
  polygon?: [number, number][] // drawn closed, e.g. class means in cyclic order
  square?: boolean // equal axis scales
  bounds?: [number, number, number, number] // xmin, xmax, ymin, ymax
  alpha?: number
}

export default function Scatter2D({ xy, colors, size = 3, width = 420, height = 420, labels, polygon, square = true, bounds, alpha = 0.8 }: Scatter2DProps) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    const ctx = c?.getContext('2d')
    if (!c || !ctx) return
    const dpr = window.devicePixelRatio || 1
    c.width = width * dpr
    c.height = height * dpr
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, width, height)
    const all = [...xy, ...(labels?.map((l) => l.xy) ?? []), ...(polygon ?? [])]
    let [x0, x1, y0, y1] = bounds ?? [Math.min(...all.map((p) => p[0])), Math.max(...all.map((p) => p[0])), Math.min(...all.map((p) => p[1])), Math.max(...all.map((p) => p[1]))]
    if (square) {
      const cx = (x0 + x1) / 2
      const cy = (y0 + y1) / 2
      const r = Math.max(x1 - x0, y1 - y0) / 2 || 1
      ;[x0, x1, y0, y1] = [cx - r, cx + r, cy - r, cy + r]
    }
    const pad = 16
    const sx = (v: number) => pad + ((v - x0) / (x1 - x0 || 1)) * (width - 2 * pad)
    const sy = (v: number) => height - pad - ((v - y0) / (y1 - y0 || 1)) * (height - 2 * pad)
    ctx.globalAlpha = alpha
    xy.forEach(([x, y], i) => {
      ctx.fillStyle = colors[i] ?? '#888'
      ctx.beginPath()
      ctx.arc(sx(x), sy(y), size, 0, 2 * Math.PI)
      ctx.fill()
    })
    ctx.globalAlpha = 1
    const fg = getComputedStyle(document.documentElement).getPropertyValue('--text').trim() || '#000'
    if (polygon && polygon.length > 1) {
      ctx.strokeStyle = fg
      ctx.globalAlpha = 0.6
      ctx.lineWidth = 1.5
      ctx.beginPath()
      polygon.forEach(([x, y], i) => (i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))))
      ctx.closePath()
      ctx.stroke()
      ctx.globalAlpha = 1
    }
    labels?.forEach((l) => {
      ctx.fillStyle = l.color ?? fg
      ctx.beginPath()
      ctx.arc(sx(l.xy[0]), sy(l.xy[1]), 7, 0, 2 * Math.PI)
      ctx.fill()
      ctx.strokeStyle = fg
      ctx.stroke()
      ctx.fillStyle = fg
      ctx.font = '12px system-ui'
      ctx.fillText(l.text, sx(l.xy[0]) + 9, sy(l.xy[1]) + 4)
    })
  }, [xy, colors, size, width, height, labels, polygon, square, bounds, alpha])
  return <canvas ref={ref} style={{ width: '100%', maxWidth: width, aspectRatio: `${width} / ${height}`, display: 'block' }} />
}
