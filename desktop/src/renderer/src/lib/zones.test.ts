import { describe, expect, it } from 'vitest'
import { DEFAULT_ZONE_PRESET } from '../../../main/transferGeometry'
import { clickPoint, fromScreen, layoutOutlines, svgPoints, toScreen } from './zones'

describe('zone outlines for the preview', () => {
  it('draws the default bands as the basket uses them: inside the bottom 35%, opening above', () => {
    const { inside, opening } = layoutOutlines({ mode: 'bands', ...DEFAULT_ZONE_PRESET })
    expect(inside.map((p) => p.y)).toEqual([0.65, 0.65, 1, 1])
    expect(opening.map((p) => p.y)).toEqual([0.45, 0.45, 0.65, 0.65])
  })

  it('shows nothing rather than throwing for a half-typed band preset', () => {
    expect(
      layoutOutlines({
        mode: 'bands',
        cartEdge: 'bottom',
        insideFraction: 0.7,
        openingFraction: 0.5
      })
    ).toEqual({ inside: [], opening: [] })
  })

  it('crosses the preview mirror both ways with one rule', () => {
    const p = { x: 0.2, y: 0.7 }
    expect(toScreen(p, true)).toEqual({ x: 0.8, y: 0.7 })
    const back = fromScreen(toScreen(p, true), true)
    expect(back.x).toBeCloseTo(p.x, 12)
    expect(back.y).toBe(p.y)
    expect(toScreen(p, false)).toBe(p)
    expect(svgPoints([p], true)).toBe('0.8000,0.7000')
  })

  it('turns a click into a clamped 0-1 point on the image', () => {
    const rect = { left: 100, top: 50, width: 200, height: 100 }
    expect(clickPoint(150, 75, rect)).toEqual({ x: 0.25, y: 0.25 })
    expect(clickPoint(400, 0, rect)).toEqual({ x: 1, y: 0 })
    expect(clickPoint(10, 10, { left: 0, top: 0, width: 0, height: 0 })).toBeNull()
  })
})
