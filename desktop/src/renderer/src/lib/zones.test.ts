import { describe, expect, it } from 'vitest'
import { DEFAULT_ZONE_PRESET } from '../../../main/transferGeometry'
import {
  clickPoint,
  commit,
  fromScreen,
  historyOf,
  layoutOutlines,
  moveCorner,
  redo,
  svgPoints,
  toScreen,
  undo
} from './zones'

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

  it('moves one corner, clamped to the picture', () => {
    const sq = [
      { x: 0.3, y: 0.5 },
      { x: 0.7, y: 0.5 },
      { x: 0.7, y: 0.9 },
      { x: 0.3, y: 0.9 }
    ]
    const moved = moveCorner(sq, 2, { x: 1.4, y: 0.95 })
    expect(moved[2]).toEqual({ x: 1, y: 0.95 })
    expect(moved[0]).toEqual(sq[0])
  })

  it('undoes and redoes whole states, and a new edit forgets the redo', () => {
    let h = historyOf(1)
    h = commit(h, 2)
    h = commit(h, 3)
    h = undo(h)
    expect(h.present).toBe(2)
    h = undo(h)
    expect(h.present).toBe(1)
    expect(undo(h)).toBe(h) // nothing further back
    h = redo(h)
    expect(h.present).toBe(2)
    h = commit(h, 9)
    expect(h.future).toEqual([])
    expect(redo(h)).toBe(h)
    // An edit that changed nothing is not a step to undo.
    expect(commit(h, 9).past).toEqual(h.past)
  })

  it('turns a click into a clamped 0-1 point on the image', () => {
    const rect = { left: 100, top: 50, width: 200, height: 100 }
    expect(clickPoint(150, 75, rect)).toEqual({ x: 0.25, y: 0.25 })
    expect(clickPoint(400, 0, rect)).toEqual({ x: 1, y: 0 })
    expect(clickPoint(10, 10, { left: 0, top: 0, width: 0, height: 0 })).toBeNull()
  })
})
