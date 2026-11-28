// Unit tests for the verdict tally (src/lib/verdicts.ts): post-hoc rows are listed but never counted.
import { describe, expect, it } from 'vitest'
import { tally } from '../src/lib/verdicts'
import type { ReplicationResult } from '../src/lib/types'

const row = (verdict: ReplicationResult['verdict'], post_hoc?: string): ReplicationResult => ({
  claim: 'c',
  published: 'p',
  ours: 'o',
  criterion: 'k',
  verdict,
  ...(post_hoc ? { post_hoc } : {}),
})

describe('tally', () => {
  it('counts pre-stated rows by verdict and post-hoc rows separately', () => {
    const t = tally([row('pass'), row('pass'), row('fail'), row('partial'), row('pass', 'criteria written late'), row('fail', 'x')])
    expect(t).toEqual({ pass: 2, partial: 1, fail: 1, na: 0, preStated: 4, postHoc: 2 })
  })
  it('handles an empty list', () => {
    expect(tally([])).toEqual({ pass: 0, partial: 0, fail: 0, na: 0, preStated: 0, postHoc: 0 })
  })
})
