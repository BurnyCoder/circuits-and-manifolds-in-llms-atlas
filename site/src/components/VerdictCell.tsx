// Verdict cell of a replication table. Post-hoc rows show their verdict plus the reason they are not counted.
import type { ReplicationResult } from '../lib/types'

export default function VerdictCell({ r }: { r: ReplicationResult }) {
  return (
    <td className={`outcome ${r.verdict}`}>
      {r.verdict}
      {r.post_hoc && <span className="posthoc">post-hoc, not counted: {r.post_hoc}</span>}
    </td>
  )
}
