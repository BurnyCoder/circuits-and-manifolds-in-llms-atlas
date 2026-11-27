// Loads all atlas entries at build time.
// Global context: Vite's import.meta.glob (https://vite.dev/guide/features.html#glob-import) inlines every
// src/content/entries/*.md file as a raw string; we split YAML frontmatter from the Markdown body and parse it with
// the `yaml` package (https://eemeli.org/yaml/). The same parser is reused by tests/content.test.ts in Node.
import { parseEntry } from './frontmatter'
import type { Entry } from './types'

// Eager raw import: { './content/entries/ioi.md': '---\n...' }.
const files = import.meta.glob('../content/entries/*.md', { query: '?raw', import: 'default', eager: true }) as Record<
  string,
  string
>

/** All entries, sorted by `order` then title. */
export const entries: Entry[] = Object.entries(files)
  .map(([path, raw]) => parseEntry(raw, path))
  .sort((a, b) => (a.order ?? 999) - (b.order ?? 999) || a.title.localeCompare(b.title))

/** Lookup table slug -> entry. */
export const entryBySlug: Record<string, Entry> = Object.fromEntries(entries.map((e) => [e.slug, e]))
