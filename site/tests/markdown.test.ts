// Unit tests for the {{template}} number formats used in entry prose (src/lib/markdown.ts).
import { describe, expect, it } from 'vitest'
import { fillTemplates, fmt } from '../src/lib/markdown'

describe('fmt', () => {
  it('formats plain numbers, percentages and integers', () => {
    expect(fmt(3.5372)).toBe('3.54')
    expect(fmt(0.929927, 'pct')).toBe('93.0%')
    expect(fmt(2100, 'int')).toBe('2,100')
    expect(fmt('6/11')).toBe('6/11')
  })
  it('uses scientific notation with a superscript exponent for large values', () => {
    expect(fmt(789449.3381, 'sci')).toBe('7.9 × 10⁵')
    expect(fmt(12.61, 'sci')).toBe('12.6') // small values fall back to the default format
  })
})

describe('fillTemplates', () => {
  it('fills known keys and marks unknown ones', () => {
    expect(fillTemplates('{{a}} and {{b|pct}}', { a: 2, b: 0.5 })).toBe('2 and 50.0%')
    expect(fillTemplates('{{missing}}', { a: 1 })).toBe('[missing:missing]')
    expect(fillTemplates('{{a}}', undefined)).toBe('…')
  })
})
