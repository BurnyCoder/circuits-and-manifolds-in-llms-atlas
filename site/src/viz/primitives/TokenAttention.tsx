// Token strip where hovering/clicking a destination token colors every source token by its attention weight.
// Mirrors CircuitsVis' attention-pattern view (https://github.com/TransformerLensOrg/CircuitsVis) in plain React.
import { useState } from 'react'

export default function TokenAttention({
  tokens,
  pattern,
  initial,
  highlight,
}: {
  tokens: string[]
  pattern: number[][] // [dest][src]
  initial?: number
  highlight?: Record<number, string> // token index -> outline color (e.g. IO / S positions)
}) {
  const [dest, setDest] = useState(initial ?? tokens.length - 1)
  const row = pattern[dest] ?? []
  const max = Math.max(...row, 1e-9)
  return (
    <div>
      <div className="token-strip" role="list">
        {tokens.map((t, i) => {
          const w = i <= dest ? row[i] ?? 0 : 0
          return (
            <span
              key={i}
              role="listitem"
              className={i === dest ? 'sel' : ''}
              title={`attention from “${tokens[dest]}” → “${t}”: ${(row[i] ?? 0).toFixed(3)}`}
              onMouseEnter={() => setDest(i)}
              onClick={() => setDest(i)}
              style={{
                background: `rgba(124, 58, 237, ${Math.min(1, w / max) * 0.85})`,
                color: w / max > 0.55 ? '#fff' : undefined,
                outline: highlight?.[i] ? `2px solid ${highlight[i]}` : undefined,
              }}
            >
              {t.replace(/\n/g, '⏎') || '·'}
            </span>
          )
        })}
      </div>
      <p className="viz-note">
        Destination token: <b>“{tokens[dest]}”</b> (position {dest}). Purple intensity = attention weight to each earlier token
        (max {max.toFixed(2)}).
      </p>
    </div>
  )
}
