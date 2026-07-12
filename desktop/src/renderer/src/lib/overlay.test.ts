import { describe, it, expect } from 'vitest'
import { boxToPixels, layoutDetections, boxToPercent } from './overlay'
import type { Detection } from './ws'

describe('boxToPercent', () => {
  it('maps a normalized box to CSS percentages relative to the image wrapper', () => {
    expect(boxToPercent([0.25, 0.1, 0.75, 0.6])).toEqual({
      left: 25,
      top: 10,
      width: 50,
      height: 50
    })
  })
})

describe('boxToPixels', () => {
  it('scales a normalized box to the displayed image size', () => {
    // [x1,y1,x2,y2] = [0.5,0.5,1,1] on a 200x100 image -> right-bottom quadrant
    expect(boxToPixels([0.5, 0.5, 1, 1], 200, 100)).toEqual({ x: 100, y: 50, w: 100, h: 50 })
  })

  it('handles a full-frame box', () => {
    expect(boxToPixels([0, 0, 1, 1], 300, 150)).toEqual({ x: 0, y: 0, w: 300, h: 150 })
  })
})

describe('layoutDetections', () => {
  it('produces pixel rects with class + confidence labels', () => {
    const dets: Detection[] = [{ track_id: 3, cls: 'banana', conf: 0.912, box: [0, 0, 0.5, 0.5] }]
    const rects = layoutDetections(dets, 100, 100)
    expect(rects).toHaveLength(1)
    expect(rects[0]).toMatchObject({ x: 0, y: 0, w: 50, h: 50 })
    expect(rects[0].label).toBe('banana 91%')
  })

  it('returns an empty array for no detections', () => {
    expect(layoutDetections([], 100, 100)).toEqual([])
  })
})
