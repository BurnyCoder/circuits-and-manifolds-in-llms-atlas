// Human-readable labels and colors for categories, fidelity levels and collections.
import type { Category, Evidence, Fidelity } from './types'

export const CATEGORY: Record<Category, { label: string; color: string; blurb: string }> = {
  circuit: { label: 'Circuit', color: 'var(--c-circuit)', blurb: 'Heads/neurons wired into an algorithm' },
  manifold: { label: 'Manifold', color: 'var(--c-manifold)', blurb: 'Curved geometry of representations' },
  neuron: { label: 'Neuron', color: 'var(--c-neuron)', blurb: 'Single units with a describable job' },
  feature: { label: 'SAE feature', color: 'var(--c-feature)', blurb: 'Dictionary-learned directions' },
  steering: { label: 'Steering', color: 'var(--c-steering)', blurb: 'Directions you can add to change behavior' },
  attribution: { label: 'Attribution graph', color: 'var(--c-attribution)', blurb: 'Feature-level causal graphs' },
  lens: { label: 'Lens', color: 'var(--c-lens)', blurb: 'Reading what layers would say' },
}

export const FIDELITY: Record<Fidelity, { label: string; icon: string; help: string }> = {
  recomputed: {
    label: 'Recomputed here',
    icon: '●',
    help: 'We ran an open model in this repository; the visualization shows our numbers.',
  },
  replicated: {
    label: 'Open replication',
    icon: '◆',
    help: 'Original model is closed; we show an open-model replication (by others or by us).',
  },
  redrawn: {
    label: 'Redrawn from paper',
    icon: '▲',
    help: 'Closed model: an interactive redraw or simulation built from the published numbers.',
  },
}

export const EVIDENCE: Record<Evidence, string> = {
  causal: 'causal (interventions)',
  correlational: 'correlational (probes/PCA)',
  mixed: 'causal + correlational',
}

export const COLLECTIONS: Record<string, string> = {
  'anthropic-tc': 'Anthropic · Transformer Circuits',
  'goodfire-geometry': 'Goodfire · Neural Geometry series',
  'goodfire-neurons': 'Goodfire · Neurons',
  distill: 'Distill · Circuits thread',
  tegmark: 'Tegmark lab · Geometry',
  redwood: 'Redwood / MATS circuits',
  toy: 'Toy models',
  steering: 'Steering & directions',
  other: 'Other',
}
