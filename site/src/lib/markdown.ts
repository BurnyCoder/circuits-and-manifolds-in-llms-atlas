// Markdown -> HTML for entry bodies, with KaTeX math and data templates.
// Global context: entry prose lives in Markdown (src/content/entries). Numbers we computed are never typed by hand:
// the body writes {{key}} and the value comes from the entry's data JSON `summary` (written by the Python pipeline),
// so re-running the pipeline updates the prose. Math uses $...$ (inline) and $$...$$ (display), rendered by KaTeX
// (https://katex.org/docs/api.html). Markdown parsing uses marked extensions (https://marked.js.org/using_pro).
import katex from 'katex'
import { Marked, type TokenizerAndRendererExtension } from 'marked'

const mathBlock: TokenizerAndRendererExtension = {
  name: 'mathBlock',
  level: 'block',
  start: (src) => src.indexOf('$$'),
  tokenizer(src) {
    const m = /^\$\$([\s\S]+?)\$\$/.exec(src)
    if (m) return { type: 'mathBlock', raw: m[0], text: m[1].trim() }
  },
  renderer: (t) => katex.renderToString(t.text as string, { displayMode: true, throwOnError: false }),
}

const mathInline: TokenizerAndRendererExtension = {
  name: 'mathInline',
  level: 'inline',
  start: (src) => src.indexOf('$'),
  tokenizer(src) {
    // `$x$` but not `$ 5` prices: require non-space right after the opening and before the closing dollar.
    const m = /^\$(?!\s)([^$\n]+?)(?<!\s)\$/.exec(src)
    if (m) return { type: 'mathInline', raw: m[0], text: m[1] }
  },
  renderer: (t) => katex.renderToString(t.text as string, { displayMode: false, throwOnError: false }),
}

const md = new Marked({ gfm: true, extensions: [mathBlock, mathInline] })
// Open external links in a new tab; citations point outside the site.
md.use({
  renderer: {
    link({ href, title, tokens }) {
      const text = this.parser.parseInline(tokens)
      const ext = /^https?:/.test(href)
      return `<a href="${href}"${title ? ` title="${title}"` : ''}${ext ? ' target="_blank" rel="noopener"' : ''}>${text}</a>`
    },
  },
})

/** Format a number for prose: integers as-is, otherwise 3 significant digits (e.g. 0.0123, 3.56, 87.4). */
export function fmt(v: number | string, mode?: string): string {
  if (typeof v === 'string') return v
  if (mode === 'pct') return `${(v * 100).toFixed(1)}%`
  if (mode === 'int') return Math.round(v).toLocaleString('en-US')
  if (Number.isInteger(v)) return v.toLocaleString('en-US')
  return Math.abs(v) >= 100 ? v.toFixed(1) : Number(v.toPrecision(3)).toString()
}

/** Replace {{key}} / {{key|pct}} with values; unknown keys render as a visible "[missing:key]" so tests catch them. */
export function fillTemplates(src: string, summary: Record<string, number | string> | undefined): string {
  return src.replace(/\{\{\s*([\w.-]+)\s*(?:\|\s*(\w+))?\s*\}\}/g, (_, key: string, mode?: string) => {
    const v = summary?.[key]
    return v === undefined ? (summary ? `[missing:${key}]` : '…') : fmt(v, mode)
  })
}

export function renderMarkdown(src: string, summary?: Record<string, number | string>): string {
  return md.parse(fillTemplates(src, summary), { async: false }) as string
}
