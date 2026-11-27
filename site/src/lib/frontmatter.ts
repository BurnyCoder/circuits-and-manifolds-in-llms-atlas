// Frontmatter parser shared by the site and the Vitest content tests (no Vite-specific APIs here).
import { parse } from 'yaml'
import type { Entry, EntryMeta } from './types'

const FRONTMATTER = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/

/** Split `---\nyaml\n---\nbody` and return a typed entry; throws with the file path if malformed. */
export function parseEntry(raw: string, path: string): Entry {
  const m = raw.match(FRONTMATTER)
  if (!m) throw new Error(`${path}: missing YAML frontmatter`)
  const meta = parse(m[1]) as EntryMeta
  if (!meta || typeof meta.slug !== 'string') throw new Error(`${path}: frontmatter needs a slug`)
  return { ...meta, citations: meta.citations ?? [], body: m[2].trim() }
}
