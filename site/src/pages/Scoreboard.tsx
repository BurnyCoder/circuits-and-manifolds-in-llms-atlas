// Replications scoreboard: every published-vs-recomputed row across entries (loaded from each entry's data JSON),
// plus the replications-by-others listed in entry frontmatter.
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { entries } from '../lib/content'
import { loadData } from '../lib/data'
import type { ReplicationResult } from '../lib/types'
import { tally } from '../lib/verdicts'
import VerdictCell from '../components/VerdictCell'

export default function Scoreboard() {
  const [rows, setRows] = useState<(ReplicationResult & { slug: string; title: string })[]>([])
  useEffect(() => {
    Promise.all(
      entries
        .flatMap((e) => [e.data, e.extra].filter(Boolean).map((f) => ({ e, f: f! })))
        .map(({ e, f }) =>
          loadData(f)
            .then((d) => (d.replication ?? []).map((r) => ({ ...r, slug: e.slug, title: e.title })))
            .catch(() => []),
        ),
    ).then((all) => setRows(all.flat()))
  }, [])
  const t = tally(rows)
  const others = entries.flatMap((e) => (e.replications ?? []).filter((r) => r.outcome !== 'ours').map((r) => ({ ...r, slug: e.slug, title: e.title })))
  return (
    <>
      <h1>Replications scoreboard</h1>
      <p className="prose">
        Every claim we re-tested, the number the original paper reports, what we measured on open models, the pass criterion
        (see <code>reports/planned-runs.md</code>), and the verdict. Failures are kept, not hidden. Rows marked{' '}
        <b>post-hoc</b> had their analysis or criterion chosen after results for that claim had been seen (or never had a
        pre-stated criterion); they stay in the table with the reason but are not counted below. Rows starting with{' '}
        <b>Confirmatory:</b> are re-tests written down and pushed before they ran, after an adversarial review, with
        stronger controls, fresh data or more seeds; they are counted. A partial verdict can also mean inconclusive: the
        95% confidence interval straddles the pass threshold.
      </p>
      <div className="kpis">
        <div className="kpi">
          <div className="v outcome pass">{t.pass}</div>
          <div className="l">pass</div>
        </div>
        <div className="kpi">
          <div className="v outcome partial">{t.partial}</div>
          <div className="l">partial</div>
        </div>
        <div className="kpi">
          <div className="v outcome fail">{t.fail}</div>
          <div className="l">fail</div>
        </div>
        <div className="kpi">
          <div className="v">{t.postHoc}</div>
          <div className="l">post-hoc (not counted)</div>
        </div>
      </div>
      <section className="panel table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Entry</th>
              <th>Claim</th>
              <th>Published</th>
              <th>Ours</th>
              <th>Criterion</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td>
                  <Link to={`/entry/${r.slug}`}>{r.title}</Link>
                </td>
                <td>{r.claim}</td>
                <td>{r.published}</td>
                <td>{r.ours}</td>
                <td>{r.criterion}</td>
                <VerdictCell r={r} />
              </tr>
            ))}
          </tbody>
        </table>
      </section>
      <h2>Replications and follow-ups by others</h2>
      <section className="panel table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Entry</th>
              <th>Who</th>
              <th>Model</th>
              <th>Outcome</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {others.map((r, i) => (
              <tr key={i}>
                <td>
                  <Link to={`/entry/${r.slug}`}>{r.title}</Link>
                </td>
                <td>
                  <a href={r.url} target="_blank" rel="noopener">
                    {r.who}
                  </a>
                </td>
                <td>{r.model}</td>
                <td className={`outcome ${r.outcome}`}>{r.outcome}</td>
                <td>{r.note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  )
}
