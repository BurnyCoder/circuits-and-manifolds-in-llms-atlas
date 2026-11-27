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
  // Five features squeezed into two dimensions: a pentagon of feature directions (Elhage et al. 2022, Fig. "toy models").
  'toy-superposition': () => (
    <>
      {[...Array(5).keys()].map((i) => {
        const a = (i / 5) * 2 * Math.PI - Math.PI / 2
        return <line key={i} x1={60} y1={42} x2={60 + 28 * Math.cos(a)} y2={42 + 28 * Math.sin(a)} stroke={rainbow(i / 5)} strokeWidth={3} strokeLinecap="round" />
      })}
      <circle cx={60} cy={42} r={3} fill="currentColor" />
    </>
  ),
  // Tokens -> intermediate concept -> answer (two-hop recall: Dallas -> Texas -> Austin).
  'attribution-graphs': (c) => {
    const nodes: [number, number, string][] = [
      [26, 66, 'Dallas'],
      [86, 66, 'capital'],
      [40, 40, 'Texas'],
      [76, 40, 'say cap.'],
      [60, 14, 'Austin'],
    ]
    const edges = [
      [0, 2],
      [1, 3],
      [2, 4],
      [3, 4],
      [0, 4],
    ]
    return (
      <>
        {edges.map(([a, b], i) => (
          <line key={i} x1={nodes[a][0]} y1={nodes[a][1]} x2={nodes[b][0]} y2={nodes[b][1]} stroke={c} strokeOpacity={i === 4 ? 0.3 : 0.75} strokeWidth={i === 4 ? 1 : 2} strokeDasharray={i === 4 ? '3 2' : undefined} />
        ))}
        {nodes.map(([x, y, t], i) => (
          <g key={i}>
            <rect x={x - 17} y={y - 7} width={34} height={14} rx={4} fill={i === 4 ? c : 'var(--panel)'} stroke={c} />
            <text x={x} y={y + 3} fontSize={7.5} textAnchor="middle" fill={i === 4 ? '#fff' : 'currentColor'}>
              {t}
            </text>
          </g>
        ))}
      </>
    )
  },
  // A few large neuron activations among many tiny ones (sparse neuron basis).
  'neuron-basis': (c) => (
    <>
      {[...Array(28).keys()].map((i) => {
        const big = [4, 11, 19, 23].includes(i)
        const h = big ? 30 + ((i * 7) % 20) : 3 + ((i * 5) % 6)
        return <rect key={i} x={10 + i * 3.6} y={70 - h} width={2.6} height={h} fill={big ? c : 'currentColor'} opacity={big ? 1 : 0.3} />
      })}
      <line x1={8} y1={70} x2={112} y2={70} stroke="currentColor" strokeOpacity={0.4} />
    </>
  ),
  // Harmless vs harmful prompt clusters separated along one direction.
  'refusal-direction': (c) => (
    <>
      {[...Array(14).keys()].map((i) => (
        <circle key={`a${i}`} cx={24 + ((i * 13) % 22)} cy={30 + ((i * 17) % 34)} r={2.6} fill="#59a14f" />
      ))}
      {[...Array(14).keys()].map((i) => (
        <circle key={`b${i}`} cx={78 + ((i * 11) % 22)} cy={22 + ((i * 19) % 34)} r={2.6} fill="#e15759" />
      ))}
      <path d="M38 64 L92 20" stroke={c} strokeWidth={2.5} markerEnd="url(#ah-ref)" />
      <defs>
        <marker id="ah-ref" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill={c} />
        </marker>
      </defs>
    </>
  ),
  // Activation addition: a contrast pair's difference added to the residual stream.
  actadd: (c) => (
    <>
      <text x={60} y={30} fontSize={12} textAnchor="middle" fill="currentColor" fontWeight={700}>
        Love − Hate
      </text>
      <line x1={14} y1={48} x2={106} y2={48} stroke="currentColor" strokeOpacity={0.35} strokeWidth={6} strokeLinecap="round" />
      <circle cx={60} cy={48} r={9} fill={c} />
      <text x={60} y={52} fontSize={13} textAnchor="middle" fill="#fff" fontWeight={700}>
        +
      </text>
      <text x={60} y={70} fontSize={8} textAnchor="middle" fill="currentColor" opacity={0.7}>
        residual stream
      </text>
    </>
  ),
  // Layer × position grid with a highlighted middle "workspace" band (J-lens readouts).
  jlens: (c) => (
    <>
      {[...Array(8).keys()].map((r) =>
        [...Array(12).keys()].map((col) => {
          const band = r >= 2 && r <= 5
          return <rect key={`${r}-${col}`} x={12 + col * 8} y={8 + r * 8} width={7} height={7} rx={1.5} fill={band ? c : 'var(--panel-2)'} opacity={band ? 0.35 + ((r * 3 + col * 5) % 7) / 10 : 1} />
        }),
      )}
    </>
  ),
  // Valence × arousal circumplex: emotion vectors arranged like human ratings.
  emotions: () => (
    <>
      <line x1={14} y1={42} x2={106} y2={42} stroke="currentColor" strokeOpacity={0.3} />
      <line x1={60} y1={6} x2={60} y2={76} stroke="currentColor" strokeOpacity={0.3} />
      {[...Array(18).keys()].map((i) => {
        const a = (i / 18) * 2 * Math.PI
        const r = 22 + ((i * 7) % 9)
        return <circle key={i} cx={60 + r * 1.4 * Math.cos(a)} cy={42 - r * Math.sin(a)} r={3} fill={`hsl(${Math.round(120 * (0.5 + 0.5 * Math.cos(a)))}, 75%, 52%)`} />
      })}
      <text x={104} y={38} fontSize={7} textAnchor="end" fill="currentColor" opacity={0.7}>
        pleasant
      </text>
    </>
  ),
  // A circle (manifold) tiled by several SAE latents, each covering an arc.
  'sae-manifolds': () => (
    <>
      {[...Array(6).keys()].map((i) => {
        const a0 = (i / 6) * 2 * Math.PI
        const a1 = a0 + (2 * Math.PI) / 6 + 0.25
        const r = 26
        return (
          <path key={i} d={`M${60 + r * Math.cos(a0)},${41 + r * Math.sin(a0)} A${r},${r} 0 0 1 ${60 + r * Math.cos(a1)},${41 + r * Math.sin(a1)}`} fill="none" stroke={rainbow(i / 6)} strokeWidth={6} strokeOpacity={0.75} />
        )
      })}
    </>
  ),
  // Block-sparse code: only a few blocks of the dictionary active.
  'block-sparse-featurizers': (c) => (
    <>
      {[...Array(6).keys()].map((b) =>
        [...Array(4).keys()].map((k) => (
          <rect key={`${b}-${k}`} x={14 + b * 16} y={16 + k * 12} width={14} height={10} rx={2} fill={b === 1 || b === 4 ? c : 'var(--panel-2)'} opacity={b === 1 || b === 4 ? 0.45 + k * 0.15 : 1} />
        )),
      )}
      <text x={60} y={74} fontSize={8} textAnchor="middle" fill="currentColor" opacity={0.7}>
        2 of 6 blocks on
      </text>
    </>
  ),
  // Steering along the curved manifold (arc) vs the straight chord between two days.
  'manifold-steering': (c) => (
    <>
      <circle cx={60} cy={44} r={28} fill="none" stroke="currentColor" strokeOpacity={0.25} />
      <path d={`M${60 + 28 * Math.cos(2.6)},${44 - 28 * Math.sin(2.6)} A28,28 0 0 1 ${60 + 28 * Math.cos(0.5)},${44 - 28 * Math.sin(0.5)}`} fill="none" stroke={c} strokeWidth={3} />
      <line x1={60 + 28 * Math.cos(2.6)} y1={44 - 28 * Math.sin(2.6)} x2={60 + 28 * Math.cos(0.5)} y2={44 - 28 * Math.sin(0.5)} stroke="currentColor" strokeOpacity={0.5} strokeDasharray="3 2" />
      <circle cx={60 + 28 * Math.cos(2.6)} cy={44 - 28 * Math.sin(2.6)} r={4} fill="#4e79a7" />
      <circle cx={60 + 28 * Math.cos(0.5)} cy={44 - 28 * Math.sin(0.5)} r={4} fill="#e15759" />
    </>
  ),
  // Addition on clocks: a mod-10 dial with two summed hands.
  'geometric-calculator': (c) => (
    <>
      <circle cx={60} cy={41} r={28} fill="none" stroke={c} strokeOpacity={0.5} />
      {[...Array(10).keys()].map((i) => {
        const a = (i / 10) * 2 * Math.PI - Math.PI / 2
        return (
          <text key={i} x={60 + 22 * Math.cos(a)} y={44 + 22 * Math.sin(a)} fontSize={7} textAnchor="middle" fill="currentColor">
            {i}
          </text>
        )
      })}
      <line x1={60} y1={41} x2={60 + 15 * Math.cos(-Math.PI / 2 + 0.6 * 2 * Math.PI)} y2={41 + 15 * Math.sin(-Math.PI / 2 + 0.6 * 2 * Math.PI)} stroke={c} strokeWidth={2.5} />
      <line x1={60} y1={41} x2={60 + 12 * Math.cos(-Math.PI / 2 + 0.3 * 2 * Math.PI)} y2={41 + 12 * Math.sin(-Math.PI / 2 + 0.3 * 2 * Math.PI)} stroke="currentColor" strokeWidth={1.5} opacity={0.6} />
    </>
  ),
  // Mountain-car valley with the car and its latent position.
  'world-model': (c) => {
    const pts = [...Array(50).keys()].map((i) => [10 + i * 2, 60 - 22 * Math.sin(3 * ((i / 49) * 1.8 - 1.2))] as const)
    return (
      <>
        <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke="currentColor" strokeOpacity={0.6} strokeWidth={2} />
        <circle cx={pts[34][0]} cy={pts[34][1] - 5} r={5} fill={c} />
        <text x={104} y={14} fontSize={10} textAnchor="end">
          ⛳
        </text>
      </>
    )
  },
  // A story's trajectory through valence × arousal space.
  'stories-in-space': (c) => {
    const pts = [
      [22, 54],
      [36, 44],
      [50, 58],
      [66, 30],
      [80, 22],
      [96, 38],
    ]
    return (
      <>
        <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke={c} strokeWidth={2} />
        {pts.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={3.5} fill={rainbow(0.6 + i / 14)} />
        ))}
        <text x={14} y={74} fontSize={8} fill="currentColor" opacity={0.7}>
          sentence by sentence
        </text>
      </>
    )
  },
  // Golden Gate feature: bridge towers and cables.
  'sae-features': () => (
    <>
      <rect x={34} y={18} width={5} height={46} fill="#c0362c" />
      <rect x={82} y={18} width={5} height={46} fill="#c0362c" />
      <path d="M8 46 Q36 14 36 20 Q60 56 84 20 Q86 14 112 46" fill="none" stroke="#c0362c" strokeWidth={2} />
      <line x1={8} y1={50} x2={112} y2={50} stroke="#c0362c" strokeWidth={3} />
      <path d="M8 66 Q30 60 60 66 T112 66" fill="none" stroke="#4e79a7" strokeOpacity={0.6} strokeWidth={2} />
    </>
  ),
  // Trait vector: an arrow pushing a persona from "neutral" to "trait".
  'persona-vectors-claude': (c) => (
    <>
      <circle cx={28} cy={36} r={8} fill="none" stroke="currentColor" strokeWidth={2} />
      <path d="M16 62 Q28 46 40 62" fill="none" stroke="currentColor" strokeWidth={2} />
      <path d="M50 42 L96 42" stroke={c} strokeWidth={3} markerEnd="url(#ah-pv)" />
      <text x={74} y={34} fontSize={8} textAnchor="middle" fill="currentColor">
        + trait
      </text>
      <defs>
        <marker id="ah-pv" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill={c} />
        </marker>
      </defs>
    </>
  ),
  // A small phylogenetic tree.
  'phylogeny-manifold': (c) => (
    <>
      <path d="M12 40 H30 M30 22 V58 M30 22 H54 M30 58 H54 M54 14 V30 M54 14 H80 M54 30 H80 M54 50 V66 M54 50 H80 M54 66 H80 M80 8 V20 M80 8 H104 M80 20 H104" fill="none" stroke={c} strokeWidth={2} />
      {[8, 20, 30, 50, 66].map((y, i) => (
        <circle key={i} cx={i < 2 ? 104 : 80} cy={y} r={3} fill={rainbow(i / 5)} />
      ))}
    </>
  ),
}
// Closed-model redraws share the schematic of their open counterpart.
THUMBS['biology-claude'] = THUMBS['attribution-graphs']
THUMBS['linebreaks-claude'] = THUMBS['linebreak-manifold']
THUMBS['scaling-monosemanticity-claude'] = THUMBS['sae-features']
THUMBS['emotions-claude'] = THUMBS['emotions']
THUMBS['global-workspace-claude'] = THUMBS['jlens']

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
