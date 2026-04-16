import { describe, it, expect, vi, beforeEach } from 'vitest'
import { findGoogleCloudCliPath, getGoogleCloudCliToken } from '../src/auth'
import { spawnSync } from 'node:child_process'

vi.mock('node:child_process', () => ({
  spawnSync: vi.fn(),
}))

describe('getGoogleCloudCliToken', () => {
  beforeEach(() => {
    vi.mocked(spawnSync).mockReset()
  })

  it('throws when no CLI is found', () => {
    vi.mocked(spawnSync).mockReturnValue({ status: 1, stdout: '', stderr: '' } as any)

    expect(() => getGoogleCloudCliToken()).toThrow('gcloud CLI not found')
  })

  it('returns token from CLI', () => {
    vi.mocked(spawnSync).mockImplementation((_cmd: string, args: string[]) => {
      if (args.includes('auth') && args.includes('print-access-token')) {
        return { status: 0, stdout: 'ya29.fake-token-12345', stderr: '' }
      }
      return { status: 0, stdout: 'Google Cloud SDK', stderr: '' }
    })

    expect(getGoogleCloudCliToken()).toBe('ya29.fake-token-12345')
  })
})

describe('findGoogleCloudCliPath', () => {
  beforeEach(() => {
    vi.mocked(spawnSync).mockReset()
  })

  it('returns first path that works', () => {
    vi.mocked(spawnSync).mockImplementation((_cmd: string, _args: string[], _opts?: any) => {
      if (_cmd.includes('/usr/local/bin/gcloud')) {
        return { status: 0, stdout: Buffer.from('Google Cloud SDK') }
      }
      return { status: 1, stdout: '', stderr: '' }
    })

    expect(findGoogleCloudCliPath()).toBe('/usr/local/bin/gcloud')
  })

  it('tries multiple paths until one succeeds', () => {
    let callCount = 0
    vi.mocked(spawnSync).mockImplementation((_cmd: string, _args: string[], _opts?: any) => {
      callCount++
      if (callCount < 3) return { status: 1, stdout: '', stderr: '' }
      return { status: 0, stdout: Buffer.from('Google Cloud SDK') }
    })

    const result = findGoogleCloudCliPath()
    expect(callCount).toBeGreaterThanOrEqual(3)
    expect(typeof result).toBe('string')
  })

  it('returns undefined when no CLI is found', () => {
    vi.mocked(spawnSync).mockImplementation(() => {
      return { status: 1, stdout: '', stderr: '' }
    })

    expect(findGoogleCloudCliPath()).toBeUndefined()
  })
})
