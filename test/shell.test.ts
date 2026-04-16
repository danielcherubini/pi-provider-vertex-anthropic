import { describe, it, expect, vi } from 'vitest'
import { spawn, exec } from '../src/shell'
import { spawnSync } from 'node:child_process'

vi.mock('node:child_process', () => ({
  spawnSync: vi.fn(),
}))

describe('spawn', () => {
  it('delegates to spawnSync with command and args', () => {
    vi.mocked(spawnSync).mockReturnValue({ status: 0, stdout: '', stderr: '' } as any)

    spawn('gcloud', ['config', 'get-value', 'project'], { stdio: 'pipe' })

    expect(spawnSync).toHaveBeenCalledWith('gcloud', ['config', 'get-value', 'project'], {
      stdio: 'pipe',
    })
  })
})

describe('exec', () => {
  it('returns trimmed stdout on success', () => {
    vi.mocked(spawnSync).mockReturnValue({
      status: 0,
      stdout: '  my-project  \n',
      stderr: '',
    } as any)

    expect(exec('gcloud', ['config', 'get-value', 'project'])).toBe('my-project')
  })

  it('throws on non-zero exit status', () => {
    vi.mocked(spawnSync).mockReturnValue({
      status: 1,
      stdout: '',
      stderr: 'ERROR: permission denied',
    } as any)

    expect(() => exec('gcloud', ['projects', 'list'])).toThrow('ERROR: permission denied')
  })

  it('throws with status code when no stderr', () => {
    vi.mocked(spawnSync).mockReturnValue({
      status: 2,
      stdout: '',
      stderr: '',
    } as any)

    expect(() => exec('gcloud', ['invalid-command'])).toThrow('Command failed with status 2')
  })

  it('passes through timeout option', () => {
    vi.mocked(spawnSync).mockReturnValue({
      status: 0,
      stdout: 'ok',
      stderr: '',
    } as any)

    exec('gcloud', ['version'], { timeout: 5000 })

    expect(spawnSync).toHaveBeenCalledWith(
      'gcloud',
      ['version'],
      expect.objectContaining({ timeout: 5000, encoding: 'utf-8', stdio: ['ignore', 'pipe', 'pipe'] }),
    )
  })
})
