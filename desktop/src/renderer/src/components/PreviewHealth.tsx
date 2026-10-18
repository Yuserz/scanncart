// The two camera-health notices, in one place because two surfaces show them: a capture whose
// picture is too dark for auto exposure to light, and a capture that has produced no box at all.
//
// They are shared rather than written out twice because the sentence *is* the product here — each
// one names the only thing an operator can do about it ("add light", "wave a hand") — and two
// copies would drift into two instructions for one condition. What each surface supplies is its own
// banner class and its own pair of ids, which is the split the resize warning already uses
// (`live-resize-mismatch` on the Live view, `model-resize-mismatch` on the Admin Panel): a test or
// a driver asserts against the surface it is looking at, so the id belongs to the surface while the
// sentence belongs to the condition.
//
// Why the Basket test tab carries them as well as the Live view: at a desk the camera showing
// nothing is the *normal* failure, and it is invisible to everything else on that screen. A tester
// can click four corners onto a black picture and save them; the tab's own state line reads
// `camera running`, which is a claim about the capture thread rather than about what the camera can
// see; and the `camera blind` pill that does exist there is a different condition — the *arrival*
// of fresh frames (`transferStream`'s 2-second rule) — which a camera delivering 30 fps of black
// never trips. So the two facts this file renders are the ones that tell a tester "the camera
// cannot see anything" apart from "your zones are wrong".

import type { JSX } from 'react'
import type { FrameMessage } from '../lib/ws'
import { describeQuietS, useAimWatch } from '../hooks/useAimWatch'

export interface PreviewHealthProps {
  /** The newest frame; its `stats.too_dark` is the camera's own verdict on the light. */
  frame: FrameMessage | null
  /** Whether a capture is running — the aim pre-flight only judges a running one. */
  running: boolean
  /** The banner class this surface's stylesheet owns. */
  className: string
  /** This surface's ids for the two notices. */
  ids: { dark: string; aim: string }
}

export function PreviewHealth({ frame, running, className, ids }: PreviewHealthProps): JSX.Element {
  // Absent on a sidecar that predates the field, and false whenever auto exposure is off - the
  // operator owns the controls then, and nothing is judging the picture.
  const tooDark = frame?.stats?.too_dark ?? false
  // The aim pre-flight: has this capture ever produced a box, and how long has it run without
  // one. The banner below turns the second answer into the one question an empty counter and a
  // curtain cannot both answer.
  const aim = useAimWatch(running, frame)

  return (
    <>
      {/* The room is too dark for the shutter this framerate allows, and auto exposure has
          stopped brightening because brightness only lifts black to grey on this camera (no gain
          over MSMF). A state, not an event: it clears by itself when the light comes back, so it
          is not dismissible. It names the fix, light, and the one trade the app can make. */}
      {tooDark && (
        <div className={className} role="status" data-testid={ids.dark}>
          <span>
            <b>Too dark for the camera at this frame rate.</b> Add light over the basket — auto
            exposure has stopped brightening the picture, because more brightness would only turn it
            grey. Or switch on <i>Allow 30 fps when dark</i> in Camera tuning.
          </span>
        </div>
      )}
      {/* The aim pre-flight (docs/POS_SMOKE_TEST.md §0 "Aim"). A capture that has produced no
          box at all cannot tell an empty counter from a camera pointed at a wall — the two are
          identical to every filter this app runs, because they are identical to the detector. So
          after the twenty seconds the smoke test already grants an empty counter, this banner
          asks the one question that separates the cases: wave a hand and watch. It is a latch on
          evidence rather than an opinion — it clears on the first box this capture produces and
          re-arms on every capture start, so a restart re-runs the check a rig that moved
          deserves. Not dismissible, and the dismissal would be worthless anyway: an operator's
          click cannot distinguish "empty counter" from "curtain", but the camera can. */}
      {running && aim.warn && (
        <div className={className} role="status" data-testid={ids.aim}>
          <span>
            <b>No box has appeared in {describeQuietS(aim.quietS)} of live video.</b> If the counter
            is empty that is normal — but prove the aim before the next customer: wave a hand where
            items go. A box should appear within a couple of seconds; nothing appearing as you wave
            means the camera is not looking at the counter. Re-aim it, then wave again.
          </span>
        </div>
      )}
    </>
  )
}
