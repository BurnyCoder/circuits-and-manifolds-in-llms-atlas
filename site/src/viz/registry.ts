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
}
