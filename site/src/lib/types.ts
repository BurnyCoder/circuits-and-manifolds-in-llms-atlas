// Shared types for atlas entries.
// Global context: every atlas entry is one Markdown file in src/content/entries/<slug>.md whose YAML frontmatter
// matches `EntryMeta`. The Markdown body holds the prose; `viz` names the interactive component (src/viz/registry.ts)
// and `data` names the JSON file produced by the Python pipeline (src/atlas/entries/<slug>.py -> public/data/).

/** What kind of mechanistic-interpretability object the entry describes. */
export type Category = 'circuit' | 'manifold' | 'neuron' | 'feature' | 'steering' | 'attribution' | 'lens'

/** How the data shown on the page was obtained. */
export type Fidelity =
  | 'recomputed' // we ran an open model in this repo and the viz shows our numbers
  | 'replicated' // open-model replication by others is shown/linked; our page redraws it
  | 'redrawn' // closed model: faithful redraw / simulation from the paper's published numbers

/** Type of evidence the original work offers for its central claim. */
export type Evidence = 'causal' | 'correlational' | 'mixed'

export interface Citation {
  id: string
  title: string
  authors: string
  year: number
  venue?: string
  url: string
  /** Optional deep link anchor or section, e.g. "#argument-phase-change" or "§3.1". */
  where?: string
}

export interface Replication {
  who: string
  url: string
  model: string
  outcome: 'replicated' | 'partial' | 'failed' | 'ours'
  note: string
}

export interface Quote {
  /** Verbatim, at most 15 words (enforced by tests/content.test.ts). */
  text: string
  /** Citation id from `citations` the quote comes from. */
  source: string
  where?: string
}

export interface EntryMeta {
  slug: string
  title: string
  category: Category
  collection: string
  models: string[]
  fidelity: Fidelity
  evidence: Evidence
  /** One-line summary for cards. */
  summary: string
  /** Key of the interactive component in src/viz/registry.ts. */
  viz?: string
  /** JSON file in public/data/ produced by the Python pipeline. */
  data?: string
  /** Display order within the grid (lower first). */
  order?: number
  quote?: Quote
  citations: Citation[]
  replications?: Replication[]
  related?: string[]
  /** Components of the Model Map this entry highlights, e.g. {model: 'gpt2-small', heads: ['9.9']}. */
  modelMap?: { model: string; heads?: string[]; layers?: number[]; mlps?: number[] }
  tags?: string[]
}

export interface Entry extends EntryMeta {
  /** Markdown body (after frontmatter). */
  body: string
}

/** Shape shared by every data JSON file written by src/atlas/common/io.py. */
export interface EntryData {
  meta: { slug: string; generated: string; git_commit?: string; models?: string[]; runtime_s?: number }
  /** Flat key -> value map used for {{key}} templates in Markdown bodies. */
  summary: Record<string, number | string>
  replication?: ReplicationResult[]
  [key: string]: unknown
}

export interface ReplicationResult {
  claim: string
  published: string
  ours: string
  criterion: string
  verdict: 'pass' | 'partial' | 'fail' | 'n/a'
}
