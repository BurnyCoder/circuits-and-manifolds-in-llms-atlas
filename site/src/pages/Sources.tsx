// Full bibliography: every citation across entries, de-duplicated by URL, with the entries that cite it.
import { Link } from 'react-router-dom'
import { entries } from '../lib/content'

export default function Sources() {
  const byUrl = new Map<string, { c: (typeof entries)[0]['citations'][0]; slugs: Set<string> }>()
  for (const e of entries)
    for (const c of e.citations) {
      const k = c.url.replace(/\/$/, '')
      if (!byUrl.has(k)) byUrl.set(k, { c, slugs: new Set() })
      byUrl.get(k)!.slugs.add(e.slug)
    }
  const list = [...byUrl.values()].sort((a, b) => a.c.authors.localeCompare(b.c.authors))
  return (
    <>
      <h1>Sources</h1>
      <p className="prose">
        {list.length} works cited across {entries.length} entries. Quotes are kept under 15 words; everything else is
        paraphrased with links to the exact section where possible.
      </p>
      <ol className="cite-list panel">
        {list.map(({ c, slugs }) => (
          <li key={c.url}>
            {c.authors} ({c.year}). <a href={c.url} target="_blank" rel="noopener">{c.title}</a>
            {c.venue ? `. ${c.venue}` : ''}. Cited in:{' '}
            {[...slugs].map((s, i) => (
              <span key={s}>
                {i ? ', ' : ''}
                <Link to={`/entry/${s}`}>{entries.find((e) => e.slug === s)?.title}</Link>
              </span>
            ))}
          </li>
        ))}
      </ol>
    </>
  )
}
