// Othello-GPT entry: step through a game and compare the true board with what a linear probe reads from the residual
// stream at each layer; probe accuracy by layer with controls; and the "edit the world model" intervention.
import { useState } from 'react'
import type { VizProps } from '../registry'
import LineChart from '../primitives/LineChart'
import { HowTo, Kpis, Tabs, divScale } from '../primitives/common'

interface Game {
  moves: number[]
  rel: number[][] // [pos][64] 0 empty, 1 mine (mover), 2 theirs
  legal: number[][]
  move_probs: number[][] // [pos][60] over tokens 1..60
  probe: Record<string, number[][]> // layer -> [pos][64] P(mine)-P(theirs)
}
interface Ex {
  board: number[]
  square: number
  legal_before: number[]
  legal_after: number[]
  probs_before: number[]
  probs_after: number[]
}
interface D {
  acc: Record<string, number[]>
  occ_acc: number[]
  games: Game[]
  show_layers: number[]
  examples: Ex[]
  summary: Record<string, number>
}

const SQUARES = [...Array(64).keys()].filter((i) => ![27, 28, 35, 36].includes(i))
const name = (s: number) => `${'ABCDEFGH'[Math.floor(s / 8)]}${(s % 8) + 1}`
const toSquareProbs = (p: number[]) => {
  const out = Array(64).fill(0)
  p.forEach((v, i) => (out[SQUARES[i]] = v))
  return out
}

function Board({
  title,
  cell = 30,
  fill,
  dots,
  outline,
  mark,
  label,
}: {
  title: string
  cell?: number
  fill: (s: number) => string
  dots?: number[] // [64] probabilities
  outline?: Set<number>
  mark?: number
  label?: (s: number) => string
}) {
  const S = cell * 8 + 22
  return (
    <figure style={{ margin: 0 }}>
      <figcaption className="viz-note" style={{ marginBottom: 4 }}>
        {title}
      </figcaption>
      <svg viewBox={`0 0 ${S} ${S}`} style={{ width: '100%', maxWidth: S }}>
        {[...Array(8).keys()].map((i) => (
          <g key={i}>
            <text x={10} y={22 + i * cell + cell / 2} fontSize={10} dominantBaseline="middle" textAnchor="middle" fill="currentColor">
              {'ABCDEFGH'[i]}
            </text>
            <text x={22 + i * cell + cell / 2} y={10} fontSize={10} textAnchor="middle" fill="currentColor">
              {i + 1}
            </text>
          </g>
        ))}
        {[...Array(64).keys()].map((s) => {
          const x = 22 + (s % 8) * cell
          const y = 22 + Math.floor(s / 8) * cell
          return (
            <g key={s}>
              <rect x={x} y={y} width={cell - 1} height={cell - 1} fill={fill(s)} stroke={outline?.has(s) ? '#16a34a' : 'none'} strokeWidth={2.5} rx={3}>
                <title>{label ? label(s) : name(s)}</title>
              </rect>
              {dots && dots[s] > 0.01 && <circle cx={x + cell / 2} cy={y + cell / 2} r={Math.sqrt(dots[s]) * cell * 0.45} fill="#f59e0b" opacity={0.9} pointerEvents="none" />}
              {mark === s && <rect x={x + 2} y={y + 2} width={cell - 5} height={cell - 5} fill="none" stroke="#ec4899" strokeWidth={2.5} rx={3} pointerEvents="none" />}
            </g>
          )
        })}
      </svg>
    </figure>
  )
}

const STATE_COLOR = ['var(--panel-2)', '#7c3aed', '#0ea5e9']

export default function Othello({ data }: VizProps) {
  const d = data as unknown as D
  const [tab, setTab] = useState<'read' | 'acc' | 'edit'>('read')
  const [gi, setGi] = useState(0)
  const [pos, setPos] = useState(20)
  const [layer, setLayer] = useState(String(d.show_layers[Math.min(3, d.show_layers.length - 1)]))
  const [ei, setEi] = useState(0)
  const g = d.games[gi]
  const probe = g.probe[layer][pos]
  const color = divScale(1)
  const legal = new Set(g.legal[pos])
  const ex = d.examples[ei]
  const layers = d.acc['mine/theirs'].map((_, i) => i)
  return (
    <div>
      <Tabs
        tabs={[
          ['read', '1 · Read the board from activations'],
          ['acc', '2 · Probe accuracy by layer'],
          ['edit', '3 · Edit the world model'],
        ]}
        value={tab}
        onChange={setTab}
      />
      {tab === 'read' && (
        <>
          <HowTo>
            Othello-GPT only ever saw move sequences like “C4 E3 F6 …”. Yet a <b>linear</b> probe recovers the full board
            from its residual stream, as long as you label squares <span style={{ color: '#7c3aed' }}>■ mine</span> /{' '}
            <span style={{ color: '#0ea5e9' }}>■ theirs</span> relative to the player who just moved. Step through the game
            and change the layer: early layers are fuzzy, middle layers read the board almost perfectly. Orange dots = the
            model's predicted next-move probabilities; green outlines = actually legal moves.
          </HowTo>
          <div className="viz-controls">
            <label>
              game
              <select className="sel" value={gi} onChange={(e) => setGi(+e.target.value)}>
                {d.games.map((_, i) => (
                  <option key={i} value={i}>
                    held-out game {i + 1}
                  </option>
                ))}
              </select>
            </label>
            <label>
              after move {pos + 1} ({name(g.moves[pos])})
              <input type="range" min={0} max={g.moves.length - 1} value={pos} onChange={(e) => setPos(+e.target.value)} />
            </label>
            <label>
              probe at layer
              <select className="sel" value={layer} onChange={(e) => setLayer(e.target.value)}>
                {d.show_layers.map((l) => (
                  <option key={l} value={l}>
                    {l === 0 ? 'embeddings (0)' : `after block ${l - 1} (${l})`}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="two-col">
            <Board title="True board (relative to the player who just moved)" fill={(s) => STATE_COLOR[g.rel[pos][s]]} mark={g.moves[pos]} />
            <Board
              title={`Linear probe readout, layer ${layer}: P(mine) − P(theirs)`}
              fill={(s) => color(probe[s])}
              dots={toSquareProbs(g.move_probs[pos])}
              outline={legal}
              label={(s) => `${name(s)}: P(mine)−P(theirs) = ${probe[s].toFixed(2)}`}
            />
          </div>
        </>
      )}
      {tab === 'acc' && (
        <>
          <HowTo>
            Accuracy of linear probes trained on {d.summary.n_games_train} games and tested on {d.summary.n_games_test}{' '}
            held-out games, for all 64 squares (3 classes). Controls: the same probe on a randomly initialised network and
            with shuffled labels; and an absolute black/white probe on the real network.
          </HowTo>
          <LineChart
            series={[
              { name: 'mine / theirs / empty (relative)', x: layers, y: d.acc['mine/theirs'], color: '#7c3aed', width: 3 },
              { name: 'black / white / empty (absolute)', x: layers, y: d.acc['black/white'], color: '#0ea5e9' },
              { name: 'random-init network (control)', x: layers, y: d.acc['random-init model'], color: '#94a3b8', dash: '5 4' },
              { name: 'shuffled labels (control)', x: layers, y: d.acc['shuffled labels'], color: '#64748b', dash: '2 3' },
              { name: 'relative probe, occupied squares only', x: layers, y: d.occ_acc, color: '#ec4899' },
            ]}
            xLabel="layer (0 = embeddings, l = after block l−1)"
            yLabel="probe accuracy"
            fmtX={(v) => String(v)}
            yDomain={[0, 1]}
          />
        </>
      )}
      {tab === 'edit' && ex && (
        <>
          <HowTo>
            Flip one disc (pink square) <i>inside the model</i> by mirroring the residual stream along that square's probe
            direction at layer {d.summary.int_layer}. If the model really uses this representation, its predicted legal
            moves should change to those of the edited board. Compare the orange dots before and after.
          </HowTo>
          <div className="viz-controls">
            {d.examples.map((_, i) => (
              <button key={i} className={`chip ${ei === i ? 'on' : ''}`} onClick={() => setEi(i)}>
                example {i + 1}
              </button>
            ))}
          </div>
          <div className="two-col">
            <Board
              title="Before: model's prediction · green = legal moves on the original board"
              fill={(s) => (ex.board[s] === 1 ? '#7c3aed' : ex.board[s] === -1 ? '#0ea5e9' : 'var(--panel-2)')}
              dots={toSquareProbs(ex.probs_before)}
              outline={new Set(ex.legal_before)}
              mark={ex.square}
            />
            <Board
              title="After flipping the pink disc in the activations · green = legal moves on the edited board"
              fill={(s) => (s === ex.square ? (ex.board[s] === 1 ? '#0ea5e9' : '#7c3aed') : ex.board[s] === 1 ? '#7c3aed' : ex.board[s] === -1 ? '#0ea5e9' : 'var(--panel-2)')}
              dots={toSquareProbs(ex.probs_after)}
              outline={new Set(ex.legal_after)}
              mark={ex.square}
            />
          </div>
          <Kpis
            items={[
              ['mean errors vs edited board — intervention', d.summary.err_int.toFixed(2)],
              ['… no intervention', d.summary.err_null.toFixed(2)],
              ['… random direction (same size)', d.summary.err_rand.toFixed(2)],
              ['test cases', d.summary.n_int],
            ]}
          />
        </>
      )}
    </div>
  )
}
