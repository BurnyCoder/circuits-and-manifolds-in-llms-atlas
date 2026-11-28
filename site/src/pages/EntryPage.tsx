// Entry page: header, interactive visualization, prose (with {{templates}} filled from data), quote, our
// replication verdicts, replications by others, citations, related entries.
import { Suspense, useMemo } from 'react'
import { Link, useParams } from 'react-router-dom'
import { CategoryTag, FidelityBadge } from '../components/Badges'
import VerdictCell from '../components/VerdictCell'
import VizBoundary from '../components/VizBoundary'
import { entryBySlug } from '../lib/content'
import { useData } from '../lib/data'
import { CATEGORY, COLLECTIONS, EVIDENCE } from '../lib/labels'
import { renderMarkdown } from '../lib/markdown'
import { tally } from '../lib/verdicts'
import { vizRegistry } from '../viz/registry'

export default function EntryPage() {
  const { slug = '' } = useParams()
  const e = entryBySlug[slug]
  const { data, error } = useData(e?.data)
  const { data: extra } = useData(e?.extra)
  // Summary for {{templates}}: the entry's own data plus the optional extra file under the "extra." prefix.
  const summary = useMemo(() => {
    if (!e?.data || !data) return undefined
    const s: Record<string, number | string> = { ...data.summary }
    if (extra) for (const [k, v] of Object.entries(extra.summary)) s[`extra.${k}`] = v
    return e.extra && !extra ? undefined : s
  }, [e, data, extra])
  const replication = [...(data?.replication ?? []), ...(extra?.replication ?? [])]
  const html = useMemo(() => (e ? renderMarkdown(e.body, summary) : ''), [e, summary])
  const t = tally(replication)
  if (!e) return <p>Unknown entry “{slug}”. <Link to="/">Back to the atlas</Link></p>

  const Viz = e.viz ? vizRegistry[e.viz] : undefined
  const cite = (id: string) => e.citations.find((c) => c.id === id)
  const q = e.quote
  const qc = q ? cite(q.source) : undefined

  return (
    <article style={{ ['--cat-color' as string]: CATEGORY[e.category].color }}>
      <p className="viz-note">
        <Link to="/">← Atlas</Link> · {COLLECTIONS[e.collection] ?? e.collection}
      </p>
      <header className="entry-head">
        <div className="meta" style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          <CategoryTag c={e.category} />
          <FidelityBadge f={e.fidelity} />
          <span className="tag">evidence: {EVIDENCE[e.evidence]}</span>
          {e.models.map((m) => (
            <span className="tag" key={m}>
              {m}
            </span>
          ))}
        </div>
        <h1>{e.title}</h1>
        <p className="summary">{e.summary}</p>
        {replication.length > 0 && (
          <p className="verdict-line">
            Our recomputation, pre-stated claims: <span className="outcome pass">{t.pass} pass</span> ·{' '}
            <span className="outcome partial">{t.partial} partial</span> · <span className="outcome fail">{t.fail} fail</span>
            {t.postHoc > 0 && <> · {t.postHoc} post-hoc row{t.postHoc > 1 ? 's' : ''} (not counted)</>} — details below.
          </p>
        )}
      </header>

      {Viz && (
        <section className="panel viz">
          {error ? (
            e.status === 'pending' ? (
              <p className="viz-note">
                ⏳ <b>Results pending.</b> This entry's full run has not finished yet. Its visualization and numbers appear
                once <code>python -m atlas.run</code> has produced <code>data/{e.data}</code>.
              </p>
            ) : (
              <p className="error">Could not load data: {error}</p>
            )
          ) : e.data && !data ? (
            <p>Loading data…</p>
          ) : (
            <VizBoundary resetKey={e.slug}>
              <Suspense fallback={<p>Loading visualization…</p>}>
                <Viz data={data} entry={e} />
              </Suspense>
            </VizBoundary>
          )}
        </section>
      )}

      {q && (
        <blockquote className="quote">
          “{q.text}”
          <cite>
            — {qc ? <a href={qc.url} target="_blank" rel="noopener">{qc.authors} ({qc.year}), {qc.title}</a> : q.source}
            {q.where ? `, ${q.where}` : ''}
          </cite>
        </blockquote>
      )}

      <section className="prose" dangerouslySetInnerHTML={{ __html: html }} />

      {replication.length > 0 && (
        <section className="panel">
          <h2>Our replication: published vs. recomputed</h2>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>Claim</th>
                  <th>Published</th>
                  <th>Ours</th>
                  <th>Criterion</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {replication.map((r, i) => (
                  <tr key={i}>
                    <td>{r.claim}</td>
                    <td>{r.published}</td>
                    <td>{r.ours}</td>
                    <td>{r.criterion}</td>
                    <VerdictCell r={r} />
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {e.replications && e.replications.length > 0 && (
        <section className="panel">
          <h2>Replications and follow-ups by others</h2>
          <ul className="repl-list">
            {e.replications.map((r) => (
              <li key={r.url + r.who}>
                <span className={`outcome ${r.outcome}`}>
                  {{ replicated: '✓ replicated', partial: '◐ partial', failed: '✗ did not replicate', ours: '● this atlas' }[r.outcome]}
                </span>
                <a href={r.url} target="_blank" rel="noopener">
                  {r.who}
                </a>{' '}
                · <i>{r.model}</i> — {r.note}
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="two-col">
        <section className="panel">
          <h2>Sources</h2>
          <ol className="cite-list">
            {e.citations.map((c) => (
              <li key={c.id} id={`cite-${c.id}`}>
                {c.authors} ({c.year}). <a href={c.url} target="_blank" rel="noopener">{c.title}</a>
                {c.venue ? `. ${c.venue}` : ''}
                {c.where ? ` — ${c.where}` : ''}
              </li>
            ))}
          </ol>
        </section>
        {e.related && e.related.length > 0 && (
          <section className="panel">
            <h2>Related entries</h2>
            <ul className="cite-list">
              {e.related
                .filter((s) => entryBySlug[s])
                .map((s) => (
                  <li key={s}>
                    <Link to={`/entry/${s}`}>{entryBySlug[s].title}</Link> — {entryBySlug[s].summary}
                  </li>
                ))}
            </ul>
          </section>
        )}
      </div>
      {data?.meta && (
        <p className="viz-note">
          Data generated {data.meta.generated}
          {data.meta.git_commit ? ` at commit ${data.meta.git_commit.slice(0, 7)}` : ''} by{' '}
          <code>src/atlas/entries/{e.slug.replace(/-/g, '_')}.py</code>.
        </p>
      )}
    </article>
  )
}
