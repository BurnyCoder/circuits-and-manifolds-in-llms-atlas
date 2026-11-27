// Error boundary around one visualization: if a viz throws while rendering (e.g. a data file with an unexpected shape),
// show a short message instead of unmounting the whole app. React only supports error boundaries as class components
// (https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary).
import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  /** Changing this key (e.g. the entry slug) clears a previous error. */
  resetKey?: string
}

export default class VizBoundary extends Component<Props, { error: Error | null; key?: string }> {
  state: { error: Error | null; key?: string } = { error: null, key: this.props.resetKey }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  static getDerivedStateFromProps(props: Props, state: { error: Error | null; key?: string }) {
    return props.resetKey !== state.key ? { error: null, key: props.resetKey } : null
  }

  componentDidCatch(error: Error) {
    console.error('visualization failed:', error)
  }

  render() {
    if (this.state.error)
      return (
        <p className="error">
          This visualization could not be drawn ({this.state.error.message}). The rest of the page is unaffected.
        </p>
      )
    return this.props.children
  }
}
