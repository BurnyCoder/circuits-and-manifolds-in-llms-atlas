// Tiny schematic thumbnails of each atlas entry's mechanism (120×80 SVG), used as "pins" in the Model Explorer and
// in finding lists. They are illustrations of the published mechanism, not data plots; clicking one in the explorer
// expands the entry's real interactive visualization.
import type { ReactNode } from 'react'
import { CATEGORY } from '../lib/labels'
import type { Category } from '../lib/types'

const W = 120
const H = 80

function Frame({ children, color, title }: { children: ReactNode; color: string; title: string }) {
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="100%" role="img" aria-label={title} style={{ display: 'block' }}>
      <defs>
        <linearGradient id={`thumb-bg-${title.replace(/\W/g, '')}`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor={color} stopOpacity={0.22} />
          <stop offset="1" stopColor={color} stopOpacity={0.06} />
        </linearGradient>
      </defs>
      <rect x={1} y={1} width={W - 2} height={H - 2} rx={10} fill={`url(#thumb-bg-${title.replace(/\W/g, '')})`} stroke={color} strokeOpacity={0.6} />
      {children}
    </svg>
  )
}

const rainbow = (t: number) => `hsl(${Math.round(t * 360)}, 80%, 58%)`

const THUMBS: Record<string, (c: string) => ReactNode> = {
  'induction-heads': (c) => (
    <>
      {['A', 'B', '…', 'A', '?'].map((t, i) => (
        <g key={i}>
          <rect x={10 + i * 21} y={48} width={17} height={17} rx={4} fill={t === '?' ? c : 'var(--panel)'} stroke={c} />
          <text x={18.5 + i * 21} y={60} fontSize={10} textAnchor="middle" fill={t === '?' ? '#fff' : 'currentColor'}>
            {t === '?' ? 'B' : t}
          </text>
        </g>
      ))}
      <path d="M95 46 C80 10, 45 10, 39 46" fill="none" stroke={c} strokeWidth={2} markerEnd="url(#ah)" />
      <path d="M81 46 C72 26, 30 26, 18 46" fill="none" stroke={c} strokeOpacity={0.45} strokeWidth={1.5} strokeDasharray="3 2" />
      <defs>
        <marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill={c} />
        </marker>
      </defs>
    </>
  ),
  ioi: () => {
    const nodes: [number, number, string][] = [
      [16, 22, '#4e79a7'],
      [16, 58, '#f28e2b'],
      [44, 58, '#e15759'],
      [64, 40, '#76b7b2'],
      [90, 18, '#59a14f'],
      [90, 40, '#edc948'],
      [90, 62, '#b07aa1'],
    ]
    const edges = [
      [1, 2],
      [2, 3],
      [0, 3],
      [3, 4],
      [3, 5],
      [3, 6],
    ]
    return (
      <>
        {edges.map(([a, b], i) => (
          <line key={i} x1={nodes[a][0]} y1={nodes[a][1]} x2={nodes[b][0]} y2={nodes[b][1]} stroke="currentColor" strokeOpacity={0.4} strokeWidth={1.5} />
        ))}
        {nodes.map(([x, y, col], i) => (
          <circle key={i} cx={x} cy={y} r={7} fill={col} />
        ))}
        <text x={108} y={44} fontSize={9} textAnchor="middle" fill="currentColor">
          Mary
        </text>
      </>
    )
  },
  'copy-suppression': (c) => (
    <>
      <text x={14} y={30} fontSize={10} fill="currentColor">
        fair in love and
      </text>
      <text x={14} y={62} fontSize={11} fill="currentColor" opacity={0.5} textDecoration="line-through">
        love
      </text>
      <text x={60} y={62} fontSize={11} fill={c} fontWeight={700}>
        war ✓
      </text>
      <path d="M90 34 C100 50, 50 50, 48 36" fill="none" stroke={c} strokeWidth={2} />
      <circle cx={30} cy={48} r={9} fill={c} />
      <text x={30} y={52} fontSize={12} textAnchor="middle" fill="#fff" fontWeight={700}>
        −
      </text>
    </>
  ),
  'greater-than': () => (
    <>
      {[...Array(10).keys()].map((r) =>
        [...Array(14).keys()].map((col) => (
          <rect key={`${r}-${col}`} x={12 + col * 7} y={8 + r * 6.4} width={7} height={6.4} fill={col / 14 > r / 10 ? `hsl(${300 - (col / 14) * 60}, 80%, ${45 + (col / 14) * 20}%)` : 'var(--panel-2)'} />
        )),
      )}
      <line x1={12} y1={8} x2={110} y2={72} stroke="#fff" strokeOpacity={0.7} strokeDasharray="3 2" />
    </>
  ),
  grokking: (c) => (
    <>
      <circle cx={60} cy={40} r={26} fill="none" stroke={c} strokeOpacity={0.4} />
      {[...Array(23).keys()].map((i) => {
        const a = (2 * Math.PI * ((i * 14) % 23)) / 23
        return <circle key={i} cx={60 + 26 * Math.cos(a)} cy={40 + 26 * Math.sin(a)} r={3} fill={rainbow(i / 23)} />
      })}
      <text x={60} y={44} fontSize={10} textAnchor="middle" fill="currentColor">
        mod 113
      </text>
    </>
  ),
  'othello-gpt': () => (
    <>
      {[...Array(36).keys()].map((i) => {
        const r = Math.floor(i / 6)
        const col = i % 6
        const v = (r * 7 + col * 3) % 5
        return <rect key={i} x={30 + col * 10} y={10 + r * 10} width={9} height={9} rx={2} fill={v === 0 ? '#0ea5e9' : v < 3 ? '#7c3aed' : 'var(--panel-2)'} />
      })}
      <text x={98} y={44} fontSize={9} textAnchor="middle" fill="currentColor">
        mine
      </text>
    </>
  ),
  'curve-detectors': () => (
    <>
      {[...Array(10).keys()].map((i) => {
        const a = (i / 10) * 2 * Math.PI
        const x = 60 + 22 * Math.cos(a)
        const y = 40 + 22 * Math.sin(a)
        return <path key={i} d={`M${x - 8 * Math.sin(a)},${y + 8 * Math.cos(a)} Q${x + 7 * Math.cos(a)},${y + 7 * Math.sin(a)} ${x + 8 * Math.sin(a)},${y - 8 * Math.cos(a)}`} fill="none" stroke={rainbow(i / 10)} strokeWidth={3} strokeLinecap="round" />
      })}
    </>
  ),
  'calendar-circles': () => (
    <>
      {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((d, i) => {
        const a = (i / 7) * 2 * Math.PI - Math.PI / 2
        return (
          <g key={i}>
            <circle cx={60 + 27 * Math.cos(a)} cy={41 + 27 * Math.sin(a)} r={7} fill={rainbow(i / 7)} />
            <text x={60 + 27 * Math.cos(a)} y={44 + 27 * Math.sin(a)} fontSize={8} textAnchor="middle" fill="#fff">
              {d}
            </text>
          </g>
        )
      })}
    </>
  ),
  'number-helix': (c) => {
    const pts = [...Array(60).keys()].map((i) => [12 + i * 1.6, 40 + 22 * Math.sin((i / 60) * 6 * Math.PI)] as const)
    return (
      <>
        <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke={c} strokeWidth={2} />
        {pts.filter((_, i) => i % 6 === 0).map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={3} fill={rainbow(i / 10)} />
        ))}
        <text x={104} y={20} fontSize={9} fill="currentColor">
          T=10
        </text>
      </>
    )
  },
  'linebreak-manifold': (c) => {
    const pts = [...Array(80).keys()].map((i) => [12 + i * 1.2, 40 + (20 * Math.sin(i / 9) + 6 * Math.sin(i / 2.2)) * (0.4 + i / 140)] as const)
    return (
      <>
        <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke={c} strokeWidth={2} />
        <text x={14} y={74} fontSize={9} fill="currentColor" opacity={0.7}>
          chars since ⏎
        </text>
      </>
    )
  },
  'belief-fractal': () => {
    const pts: [number, number][] = []
    let x = 0.3,
      y = 0.3
    const V = [
      [60, 8],
      [14, 72],
      [106, 72],
    ]
    for (let i = 0; i < 700; i++) {
      const v = V[(i * 7 + (i >> 3)) % 3]
      x = x + (v[0] - x) * 0.55
      y = y + (v[1] - y) * 0.55
      if (i > 10) pts.push([x, y])
    }
    return (
      <>
        {pts.map(([px, py], i) => (
          <circle key={i} cx={px} cy={py} r={0.9} fill={`rgb(${Math.round((py / 80) * 255)}, ${Math.round((px / 120) * 200)}, 220)`} />
        ))}
      </>
    )
  },
  'colour-bicone': () => (
    <>
      {[...Array(24).keys()].map((i) => {
        const a = (i / 24) * 2 * Math.PI
        return <circle key={i} cx={60 + 30 * Math.cos(a)} cy={40 + 11 * Math.sin(a)} r={4} fill={rainbow(i / 24)} />
      })}
      <line x1={60} y1={6} x2={60} y2={74} stroke="currentColor" strokeOpacity={0.4} />
      <circle cx={60} cy={8} r={4} fill="#fff" stroke="#888" />
      <circle cx={60} cy={72} r={4} fill="#000" />
    </>
  ),
}

/** Fallback thumbnail: the category label in a tinted card. */
function Fallback(color: string, label: string) {
  return (
    <text x={60} y={44} fontSize={11} textAnchor="middle" fill={color} fontWeight={700}>
      {label}
    </text>
  )
}

export default function Thumb({ slug, category, title }: { slug: string; category: Category; title: string }) {
  const color = CATEGORY[category].color
  const draw = THUMBS[slug]
  return (
    <Frame color={color} title={title}>
      {draw ? draw(color) : Fallback(color, CATEGORY[category].label)}
    </Frame>
  )
}
