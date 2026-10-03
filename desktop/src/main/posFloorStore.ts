// Persisted cumulative floor for the bound POS session (D2 suspended): the largest quantity ever
// posted per class. The orchestrator's floor is memory-only by nature, so an app restart
// mid-session would drop it — and the next snapshot, derived from a camera that no longer sees
// the items, would reconcile the customer's verified rows out of the cart. The file is keyed by
// `session_ref`, so only the same session restores it; a new session never inherits a previous
// customer's counts, and a finished session's floor is cleared rather than left behind.
//
// `pos-floor.json` lives beside `pos.json`; the store takes the directory rather than importing
// `app`, so it is unit-testable against a temp dir (the same shape as `posConfig.ts`).

import { promises as fs } from 'fs'
import { join } from 'path'

import type { PosFloor } from './posSession'

export const POS_FLOOR_FILENAME = 'pos-floor.json'

export class PosFloorStore {
  constructor(private readonly dir: string) {}

  private get path(): string {
    return join(this.dir, POS_FLOOR_FILENAME)
  }

  /** A missing or corrupt file reads as nothing saved; it must never crash the sync loop. */
  async load(): Promise<PosFloor | null> {
    try {
      const raw = JSON.parse(await fs.readFile(this.path, 'utf8')) as Partial<PosFloor>
      if (typeof raw.sessionRef !== 'string' || !Array.isArray(raw.items)) return null
      const items = raw.items.filter(
        (item) =>
          item != null &&
          typeof item.className === 'string' &&
          item.className.length > 0 &&
          Number.isFinite(item.quantity) &&
          item.quantity > 0 &&
          Number.isFinite(item.maxConfidence)
      )
      if (items.length === 0) return null
      return { sessionRef: raw.sessionRef, items }
    } catch {
      return null
    }
  }

  /** Atomic write (temp file + rename): a half-written floor must not read as nothing. */
  async save(floor: PosFloor): Promise<void> {
    await fs.mkdir(this.dir, { recursive: true })
    const tmp = `${this.path}.tmp`
    await fs.writeFile(tmp, `${JSON.stringify(floor, null, 2)}\n`, 'utf8')
    await fs.rename(tmp, this.path)
  }

  async clear(): Promise<void> {
    await fs.rm(this.path, { force: true })
  }
}
