// Fetches an entry's data JSON from public/data/ (relative to Vite's BASE_URL so it works on GitHub Pages).
import { useEffect, useState } from 'react'
import type { EntryData } from './types'

const cache = new Map<string, Promise<EntryData>>()

export function loadData(file: string): Promise<EntryData> {
  if (!cache.has(file)) {
    // cache: 'no-cache' revalidates with the server (ETag) on each page load, so re-generated results show up at once
    // instead of a stale HTTP-cached copy (https://developer.mozilla.org/en-US/docs/Web/API/Request/cache).
    const p = fetch(`${import.meta.env.BASE_URL}data/${file}`, { cache: 'no-cache' }).then((r) => {
      if (!r.ok) throw new Error(`data/${file}: HTTP ${r.status}`)
      return r.json() as Promise<EntryData>
    })
    cache.set(file, p)
  }
  return cache.get(file)!
}

/**
 * React hook: {data, error} for a data file (undefined file -> nothing loaded).
 * The state remembers which file it belongs to: when the file changes (navigating from one entry page to another), the
 * previous entry's data is never returned for the new file — otherwise the new visualization renders once with the old
 * entry's payload and crashes.
 */
export function useData(file?: string): { data?: EntryData; error?: string } {
  const [state, setState] = useState<{ file?: string; data?: EntryData; error?: string }>({})
  useEffect(() => {
    if (!file) return
    let alive = true
    loadData(file)
      .then((data) => alive && setState({ file, data }))
      .catch((e: Error) => alive && setState({ file, error: e.message }))
    return () => {
      alive = false
    }
  }, [file])
  return state.file === file ? { data: state.data, error: state.error } : {}
}
