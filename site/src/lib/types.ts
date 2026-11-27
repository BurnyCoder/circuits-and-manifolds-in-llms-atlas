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

/**
 * A place inside a model (ids in src/lib/models.ts).
 * part: head (layer+head) · attn (all heads of a layer) · mlp (layer) · resid (residual stream after `layer`, or a
 * `layers` range) · embed · unembed · stage (CNN/pipeline stage `name`) · band (closed models: `depth` range in %).
 */
export interface Location {
  model: string
  part: 'head' | 'attn' | 'mlp' | 'resid' | 'embed' | 'unembed' | 'stage' | 'band'
  layer?: number
  head?: number
  layers?: [number, number]
  name?: string
  depth?: [number, number]
  label: string
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
  /** Optional second data file (e.g. companion runs); its summary keys are available as {{extra.key}}. */
  extra?: string
  /** Display order within the grid (lower first). */
  order?: number
  quote?: Quote
  citations: Citation[]
  replications?: Replication[]
  related?: string[]
  /** Where in which model the finding lives; drives the clickable Model Explorer (src/components/ModelExplorer). */
  locations?: Location[]
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
