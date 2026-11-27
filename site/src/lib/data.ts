// Fetches an entry's data JSON from public/data/ (relative to Vite's BASE_URL so it works on GitHub Pages).
import { useEffect, useState } from 'react'
import type { EntryData } from './types'

const cache = new Map<string, Promise<EntryData>>()

export function loadData(file: string): Promise<EntryData> {
  if (!cache.has(file)) {
    const p = fetch(`${import.meta.env.BASE_URL}data/${file}`).then((r) => {
      if (!r.ok) throw new Error(`data/${file}: HTTP ${r.status}`)
      return r.json() as Promise<EntryData>
    })
    cache.set(file, p)
  }
  return cache.get(file)!
}

/** React hook: {data, error} for a data file (undefined file -> nothing loaded). */
export function useData(file?: string): { data?: EntryData; error?: string } {
  const [state, setState] = useState<{ data?: EntryData; error?: string }>({})
  useEffect(() => {
    if (!file) return
    let alive = true
    loadData(file)
      .then((data) => alive && setState({ data }))
      .catch((e: Error) => alive && setState({ error: e.message }))
    return () => {
      alive = false
    }
  }, [file])
  return state
}
