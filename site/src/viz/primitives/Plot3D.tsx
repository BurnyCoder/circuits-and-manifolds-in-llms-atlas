// Rotatable 3-D scatter/line plot. Plotly (https://plotly.com/javascript/3d-scatter-plots/) is imported lazily so it
// only downloads when an entry with a 3-D view is opened. Theme-aware: text/grid colors follow the CSS variables.
import { useEffect, useRef } from 'react'
import { cssVar } from './common'

export interface Trace3D {
  x: number[]
  y: number[]
  z: number[]
  mode?: 'markers' | 'lines' | 'lines+markers'
  color?: string | string[]
  size?: number
  text?: string[]
  name?: string
  width?: number
}

export default function Plot3D({ traces, height = 420, axes = ['PC1', 'PC2', 'PC3'] }: { traces: Trace3D[]; height?: number; axes?: [string, string, string] | string[] }) {
  const el = useRef<HTMLDivElement>(null)
  useEffect(() => {
    let alive = true
    import('plotly.js-dist-min').then((mod) => {
      const Plotly = (mod as unknown as { default: typeof import('plotly.js-dist-min') }).default ?? mod
      if (!alive || !el.current) return
      const text = cssVar('--text')
      const grid = cssVar('--border')
      const axis = (title: string) => ({ title: { text: title }, color: text, gridcolor: grid, zerolinecolor: grid, showbackground: false })
      Plotly.react(
        el.current,
        traces.map((t) => ({
          type: 'scatter3d',
          mode: t.mode ?? 'markers',
          x: t.x,
          y: t.y,
          z: t.z,
          text: t.text,
          name: t.name,
          hoverinfo: t.text ? 'text' : 'x+y+z',
          marker: { size: t.size ?? 4, color: t.color, line: { width: 0 } },
          line: { color: typeof t.color === 'string' ? t.color : undefined, width: t.width ?? 4 },
        })) as Plotly.Data[],
        {
          height,
          margin: { l: 0, r: 0, t: 0, b: 0 },
          paper_bgcolor: 'rgba(0,0,0,0)',
          font: { color: text },
          showlegend: traces.length > 1 && traces.some((t) => t.name),
          legend: { orientation: 'h' },
          scene: { xaxis: axis(axes[0]), yaxis: axis(axes[1]), zaxis: axis(axes[2]) },
        },
        { displaylogo: false, responsive: true },
      )
    })
    return () => {
      alive = false
    }
  }, [traces, height, axes])
  return <div ref={el} style={{ width: '100%', minHeight: height }} />
}
