// @vitest-environment node
import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { promises as fs } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'
import {
  DEFAULT_POS_CONFIG,
  PosConfigError,
  PosConfigStore,
  insecureBaseUrlWarning,
  isPosEnabled
} from './posConfig'

let dir: string
let store: PosConfigStore

beforeEach(async () => {
  dir = await fs.mkdtemp(join(tmpdir(), 'pos-config-'))
  store = new PosConfigStore(dir)
})

afterEach(async () => {
  await fs.rm(dir, { recursive: true, force: true })
})

describe('PosConfigStore', () => {
  it('returns defaults when nothing has been written', async () => {
    expect(await store.load()).toEqual(DEFAULT_POS_CONFIG)
  })

  it('round-trips a saved config', async () => {
    await store.save({
      posBaseUrl: 'http://192.168.1.20:3000',
      posSecret: 'abc',
      stationId: 'counter-1'
    })

    const loaded = await store.load()
    expect(loaded.posBaseUrl).toBe('http://192.168.1.20:3000')
    expect(loaded.posSecret).toBe('abc')
    expect(loaded.stationId).toBe('counter-1')
    // Untouched fields keep their defaults.
    expect(loaded.commitDwellS).toBe(DEFAULT_POS_CONFIG.commitDwellS)
  })

  it('falls back to defaults on a corrupt file rather than crashing', async () => {
    await fs.writeFile(join(dir, 'pos.json'), '{ not json', 'utf8')
    expect(await store.load()).toEqual(DEFAULT_POS_CONFIG)
  })

  it('drops out-of-range fields back to their defaults', async () => {
    await fs.writeFile(
      join(dir, 'pos.json'),
      JSON.stringify({ minCommitConf: 42, commitDwellS: -1, removeSettleS: 4 }),
      'utf8'
    )
    const loaded = await store.load()
    expect(loaded.minCommitConf).toBe(DEFAULT_POS_CONFIG.minCommitConf)
    expect(loaded.commitDwellS).toBe(DEFAULT_POS_CONFIG.commitDwellS)
    // A valid value in the same file still lands.
    expect(loaded.removeSettleS).toBe(4)
  })

  it('refuses a commitDwellS that does not exceed track_expiry_s', async () => {
    await expect(store.save({ commitDwellS: 1.5 }, 1.5)).rejects.toBeInstanceOf(PosConfigError)
    // Nothing was written.
    expect(await store.load()).toEqual(DEFAULT_POS_CONFIG)
  })

  it('accepts a commitDwellS above the sidecar expiry', async () => {
    const saved = await store.save({ commitDwellS: 2 }, 1.5)
    expect(saved.commitDwellS).toBe(2)
    expect((await store.load()).commitDwellS).toBe(2)
  })
})

describe('isPosEnabled', () => {
  it('is off until base URL, secret and station id are all set', () => {
    expect(isPosEnabled(DEFAULT_POS_CONFIG)).toBe(false)
    expect(isPosEnabled({ ...DEFAULT_POS_CONFIG, posBaseUrl: 'http://x' })).toBe(false)
    expect(isPosEnabled({ ...DEFAULT_POS_CONFIG, posBaseUrl: 'http://x', posSecret: 's' })).toBe(
      false
    )
    expect(
      isPosEnabled({
        ...DEFAULT_POS_CONFIG,
        posBaseUrl: 'http://x',
        posSecret: 's',
        stationId: 'counter-1'
      })
    ).toBe(true)
  })
})

describe('insecureBaseUrlWarning', () => {
  it('stays quiet for https', () => {
    expect(insecureBaseUrlWarning('https://pushcart.example.com')).toBeNull()
  })

  it('stays quiet for http on localhost', () => {
    expect(insecureBaseUrlWarning('http://localhost:3000')).toBeNull()
    expect(insecureBaseUrlWarning('http://127.0.0.1:3000')).toBeNull()
  })

  it('warns for plain http on a LAN address (the TLS rule, §2.1)', () => {
    expect(insecureBaseUrlWarning('http://192.168.1.20:3000')).toMatch(/https/)
  })

  it('reports an unparseable URL', () => {
    expect(insecureBaseUrlWarning('not a url')).not.toBeNull()
  })

  it('says nothing when the feature is unconfigured', () => {
    expect(insecureBaseUrlWarning('')).toBeNull()
  })
})
