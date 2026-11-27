// Small inline badges reused by cards and entry pages.
import { CATEGORY, FIDELITY } from '../lib/labels'
import type { Category, Fidelity } from '../lib/types'

export function CategoryTag({ c }: { c: Category }) {
  return (
    <span className="tag cat" style={{ ['--cat-color' as string]: CATEGORY[c].color }}>
      {CATEGORY[c].label}
    </span>
  )
}

export function FidelityBadge({ f }: { f: Fidelity }) {
  return (
    <span className={`badge ${f}`} title={FIDELITY[f].help}>
      {FIDELITY[f].icon} {FIDELITY[f].label}
    </span>
  )
}
