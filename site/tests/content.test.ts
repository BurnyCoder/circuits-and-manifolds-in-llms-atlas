// Content tests: every entry has valid frontmatter, ≥1 citation with an http(s) URL, a quote of ≤15 words whose
// source is one of its citations, a registered viz, an existing data file, and templates that resolve against it.
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { parseEntry } from '../src/lib/frontmatter'
import { fillTemplates } from '../src/lib/markdown'

const dir = join(__dirname, '..', 'src', 'content', 'entries')
const dataDir = join(__dirname, '..', 'public', 'data')
const registry = readFileSync(join(__dirname, '..', 'src', 'viz', 'registry.ts'), 'utf8')
const files = readdirSync(dir).filter((f) => f.endsWith('.md'))
const CATS = ['circuit', 'manifold', 'neuron', 'feature', 'steering', 'attribution', 'lens']

describe('atlas entries', () => {
  it('exist', () => expect(files.length).toBeGreaterThan(0))
  const slugs = new Set<string>()
  for (const f of files) {
    const e = parseEntry(readFileSync(join(dir, f), 'utf8'), f)
    describe(e.slug, () => {
      it('slug matches filename and is unique', () => {
        expect(f).toBe(`${e.slug}.md`)
        expect(slugs.has(e.slug)).toBe(false)
        slugs.add(e.slug)
      })
      it('has required fields', () => {
        expect(e.title).toBeTruthy()
        expect(e.summary).toBeTruthy()
        expect(CATS).toContain(e.category)
        expect(['recomputed', 'replicated', 'redrawn']).toContain(e.fidelity)
        expect(['causal', 'correlational', 'mixed']).toContain(e.evidence)
        expect(e.models.length).toBeGreaterThan(0)
      })
      it('cites sources with URLs', () => {
        expect(e.citations.length).toBeGreaterThan(0)
        for (const c of e.citations) expect(c.url).toMatch(/^https?:\/\//)
        for (const r of e.replications ?? []) expect(r.url).toMatch(/^https?:\/\//)
      })
      it('quote is short and attributed', () => {
        if (!e.quote) return
        expect(e.quote.text.split(/\s+/).length).toBeLessThanOrEqual(15)
        expect(e.citations.map((c) => c.id)).toContain(e.quote.source)
      })
      it('viz is registered', () => {
        if (e.viz) expect(registry).toMatch(new RegExp(`['"]?${e.viz}['"]?\\s*:`))
      })
      it('data file exists and templates resolve', () => {
        const tpl = e.body.match(/\{\{[^}]+\}\}/g) ?? []
        if (!e.data) {
          expect(tpl.length).toBe(0)
          return
        }
        const path = join(dataDir, e.data)
        expect(existsSync(path)).toBe(true)
        const summary = JSON.parse(readFileSync(path, 'utf8')).summary
        if (e.extra) {
          expect(existsSync(join(dataDir, e.extra))).toBe(true)
          const ex = JSON.parse(readFileSync(join(dataDir, e.extra), 'utf8')).summary
          for (const [k, v] of Object.entries(ex)) summary[`extra.${k}`] = v
        }
        expect(fillTemplates(e.body, summary)).not.toContain('[missing:')
      })
    })
  }
})
