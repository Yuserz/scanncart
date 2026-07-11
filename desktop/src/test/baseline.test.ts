import { describe, it, expect } from 'vitest'

// Placeholder to confirm the vitest + jsdom baseline is green before layering
// on real renderer/main code. Removed once real suites exist.
describe('baseline', () => {
  it('runs vitest', () => {
    expect(1 + 1).toBe(2)
  })
})
