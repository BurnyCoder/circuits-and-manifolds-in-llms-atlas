// One card in the atlas grid: click -> entry page.
import { Link } from 'react-router-dom'
import { CATEGORY } from '../lib/labels'
import type { Entry } from '../lib/types'
import { CategoryTag, FidelityBadge } from './Badges'

export default function EntryCard({ e }: { e: Entry }) {
  return (
    <Link to={`/entry/${e.slug}`} className="card" style={{ ['--cat-color' as string]: CATEGORY[e.category].color }}>
      <h3>{e.title}</h3>
      <p>{e.summary}</p>
      <div className="meta">
        <CategoryTag c={e.category} />
        <FidelityBadge f={e.fidelity} />
        {e.models.slice(0, 2).map((m) => (
          <span className="tag" key={m}>
            {m}
          </span>
        ))}
      </div>
    </Link>
  )
}
