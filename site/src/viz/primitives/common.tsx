// Shared viz helpers: tooltip hook, tabs, KPI tiles, color scales.
// Color choices: sequential data uses perceptually uniform d3 interpolators (viridis/magma/cividis); signed data uses
// diverging RdBu centered on 0 (https://d3js.org/d3-scale-chromatic). Never rainbow for magnitudes.
import * as d3 from 'd3'
import { useCallback, useState, type ReactNode } from 'react'

export interface Tip {
  x: number
  y: number
  html: ReactNode
}

/** Fixed-position tooltip; call show(event, content) / hide(). Render {node} once in the component. */
export function useTooltip() {
  const [tip, setTip] = useState<Tip | null>(null)
  const show = useCallback((e: { clientX: number; clientY: number }, html: ReactNode) => {
    setTip({ x: e.clientX + 14, y: e.clientY + 14, html })
  }, [])
  const hide = useCallback(() => setTip(null), [])
  const node = tip ? (
    <div className="tooltip" style={{ left: Math.min(tip.x, window.innerWidth - 330), top: tip.y }}>
      {tip.html}
    </div>
  ) : null
  return { show, hide, node }
}

export function Tabs<T extends string>({ tabs, value, onChange }: { tabs: [T, string][]; value: T; onChange: (t: T) => void }) {
  return (
    <div className="tabs" role="tablist">
      {tabs.map(([k, label]) => (
        <button key={k} role="tab" aria-selected={value === k} className={value === k ? 'on' : ''} onClick={() => onChange(k)}>
          {label}
        </button>
      ))}
    </div>
  )
}

export function Kpis({ items }: { items: [string, ReactNode][] }) {
  return (
    <div className="kpis">
      {items.map(([l, v]) => (
        <div className="kpi" key={l}>
          <div className="v">{v}</div>
          <div className="l">{l}</div>
        </div>
      ))}
    </div>
  )
}

export function HowTo({ children }: { children: ReactNode }) {
  return <div className="howto">💡 {children}</div>
}

/** Sequential scale 0..max (viridis). */
export const seqScale = (max: number, interp = d3.interpolateViridis) => d3.scaleSequential(interp).domain([0, max || 1])

/** Diverging scale centered at 0, symmetric (RdBu reversed so positive = red). */
export const divScale = (absMax: number) => d3.scaleDiverging((t: number) => d3.interpolateRdBu(1 - t)).domain([-absMax || -1, 0, absMax || 1])

/** Categorical palette (Tableau10 is reasonably colorblind-tolerant for ≤8 classes). */
export const cat10 = d3.schemeTableau10

export const fmt = (v: number, d = 2) => (Math.abs(v) >= 1000 ? d3.format(',.0f')(v) : v.toFixed(d))

/** Read a CSS custom property (theme-aware colors for D3-drawn axes). */
export const cssVar = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888'
