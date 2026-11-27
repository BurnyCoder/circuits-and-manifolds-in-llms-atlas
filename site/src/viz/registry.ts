// Maps the `viz:` key of an entry's frontmatter to a lazily loaded React component.
// Lazy loading (https://react.dev/reference/react/lazy) keeps the home page light: Plotly and heavy entry
// visualizations only download when that entry is opened.
import { lazy, type ComponentType, type LazyExoticComponent } from 'react'
import type { Entry, EntryData } from '../lib/types'

export interface VizProps {
  data?: EntryData
  entry: Entry
}

type LazyViz = LazyExoticComponent<ComponentType<VizProps>>

export const vizRegistry: Record<string, LazyViz> = {
  induction: lazy(() => import('./entries/Induction')),
  ioi: lazy(() => import('./entries/IOI')),
  grokking: lazy(() => import('./entries/Grokking')),
  'copy-suppression': lazy(() => import('./entries/CopySuppression')),
  'greater-than': lazy(() => import('./entries/GreaterThan')),
  othello: lazy(() => import('./entries/Othello')),
  'curve-detectors': lazy(() => import('./entries/CurveDetectors')),
  'calendar-circles': lazy(() => import('./entries/CalendarCircles')),
  'number-helix': lazy(() => import('./entries/NumberHelix')),
  linebreak: lazy(() => import('./entries/Linebreak')),
  'belief-fractal': lazy(() => import('./entries/BeliefFractal')),
  'colour-bicone': lazy(() => import('./entries/ColourBicone')),
  'toy-superposition': lazy(() => import('./entries/ToySuperposition')),
  'attribution-graph': lazy(() => import('./entries/AttributionGraph')),
  'neuron-basis': lazy(() => import('./entries/NeuronBasis')),
  refusal: lazy(() => import('./entries/Refusal')),
  actadd: lazy(() => import('./entries/ActAdd')),
  jlens: lazy(() => import('./entries/JLens')),
  emotions: lazy(() => import('./entries/Emotions')),
  'sae-manifolds': lazy(() => import('./entries/SaeManifolds')),
  'block-sparse': lazy(() => import('./entries/BlockSparse')),
  'manifold-steering': lazy(() => import('./entries/ManifoldSteering')),
  'geometric-calculator': lazy(() => import('./entries/GeometricCalculator')),
  'world-model': lazy(() => import('./entries/WorldModel')),
  stories: lazy(() => import('./entries/Stories')),
  'sae-features': lazy(() => import('./entries/SaeFeatures')),
  'biology-claude': lazy(() => import('./entries/BiologyClaude')),
  'linebreaks-claude': lazy(() => import('./entries/LinebreaksClaude')),
  'scaling-mono': lazy(() => import('./entries/Redraws').then((m) => ({ default: m.ScalingMono }))),
  'emotions-claude': lazy(() => import('./entries/Redraws').then((m) => ({ default: m.EmotionsClaude }))),
  'workspace-claude': lazy(() => import('./entries/Redraws').then((m) => ({ default: m.WorkspaceClaude }))),
  'persona-vectors': lazy(() => import('./entries/Redraws').then((m) => ({ default: m.PersonaClaude }))),
  phylogeny: lazy(() => import('./entries/Redraws').then((m) => ({ default: m.Phylogeny }))),
}
