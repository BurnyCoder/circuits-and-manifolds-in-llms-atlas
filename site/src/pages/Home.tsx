// Home: hero, the interactive Model Map ("where circuits live"), filters, and the grid of entries.
import { useMemo, useState } from 'react'
import EntryCard from '../components/EntryCard'
import ModelMap from '../components/ModelMap'
import { entries } from '../lib/content'
import { CATEGORY, COLLECTIONS, FIDELITY } from '../lib/labels'
import type { Category, Fidelity } from '../lib/types'

export default function Home() {
  const [cats, setCats] = useState<Set<Category>>(new Set())
  const [fid, setFid] = useState<Fidelity | ''>('')
  const [coll, setColl] = useState('')
  const [q, setQ] = useState('')

  const shown = useMemo(() => {
    const needle = q.trim().toLowerCase()
    return entries.filter(
      (e) =>
        (cats.size === 0 || cats.has(e.category)) &&
        (!fid || e.fidelity === fid) &&
        (!coll || e.collection === coll) &&
        (!needle ||
          [e.title, e.summary, ...e.models, ...(e.tags ?? [])].join(' ').toLowerCase().includes(needle)),
    )
  }, [cats, fid, coll, q])

  const toggle = (c: Category) =>
    setCats((s) => {
      const n = new Set(s)
      if (n.has(c)) n.delete(c)
      else n.add(c)
      return n
    })

  const collections = [...new Set(entries.map((e) => e.collection))]

  return (
    <>
      <section className="hero">
        <h1>Atlas of Circuits &amp; Manifolds</h1>
        <p>
          A field guide to the mechanisms researchers have found inside neural networks: attention-head circuits,
          curved representation manifolds, interpretable neurons, sparse-autoencoder features, steering vectors,
          attribution graphs and lenses. Each entry explains the finding in plain words and in technical detail, links
          every claim to its source, and lets you explore an interactive visualization. Where the model is open, the
          data was <b>recomputed in this repository</b>; closed-model results are faithful redraws, shown next to
          open-model replications.
        </p>
      </section>

      <ModelMap />

      <div className="filters" role="group" aria-label="Filters">
        {(Object.keys(CATEGORY) as Category[]).map((c) => (
          <button
            key={c}
            className={`chip ${cats.has(c) ? 'on' : ''}`}
            style={{ ['--chip-color' as string]: CATEGORY[c].color }}
            onClick={() => toggle(c)}
            title={CATEGORY[c].blurb}
          >
            {CATEGORY[c].label}
          </button>
        ))}
        <select className="sel" value={fid} onChange={(e) => setFid(e.target.value as Fidelity | '')}>
          <option value="">All data sources</option>
          {(Object.keys(FIDELITY) as Fidelity[]).map((f) => (
            <option key={f} value={f}>
              {FIDELITY[f].label}
            </option>
          ))}
        </select>
        <select className="sel" value={coll} onChange={(e) => setColl(e.target.value)}>
          <option value="">All collections</option>
          {collections.map((c) => (
            <option key={c} value={c}>
              {COLLECTIONS[c] ?? c}
            </option>
          ))}
        </select>
        <input className="search" placeholder="Search entries, models…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      <p className="viz-note">
        {shown.length} of {entries.length} entries
      </p>
      <div className="grid">
        {shown.map((e) => (
          <EntryCard key={e.slug} e={e} />
        ))}
      </div>
    </>
  )
}
