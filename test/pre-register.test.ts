import { describe, it, expect } from 'vitest'
import { collectPreRegisterModels } from '../src/pre-register'

describe('collectPreRegisterModels', () => {
  it('extracts model IDs from enabledModels with vertex-anthropic/ prefix', () => {
    const settings = {
      enabledModels: [
        'vertex-anthropic/claude-opus-4-6',
        'vertex-anthropic/claude-haiku-4-5@20251001',
        'koji/some-model',
      ],
    }
    const result = collectPreRegisterModels(settings)
    expect(result).toEqual(['claude-opus-4-6', 'claude-haiku-4-5@20251001'])
  })

  it('includes defaultModel when defaultProvider is vertex-anthropic', () => {
    const settings = {
      defaultProvider: 'vertex-anthropic',
      defaultModel: 'claude-opus-4-6',
      enabledModels: [],
    }
    const result = collectPreRegisterModels(settings)
    expect(result).toEqual(['claude-opus-4-6'])
  })

  it('deduplicates when model is in both enabledModels and defaultModel', () => {
    const settings = {
      defaultProvider: 'vertex-anthropic',
      defaultModel: 'claude-opus-4-6',
      enabledModels: ['vertex-anthropic/claude-opus-4-6'],
    }
    const result = collectPreRegisterModels(settings)
    expect(result).toEqual(['claude-opus-4-6'])
  })

  it('returns empty array for null settings', () => {
    expect(collectPreRegisterModels(null)).toEqual([])
  })

  it('returns empty array for non-object settings', () => {
    expect(collectPreRegisterModels('string')).toEqual([])
  })

  it('ignores non-string entries in enabledModels', () => {
    const settings = {
      enabledModels: [123, null, undefined, 'vertex-anthropic/claude-opus-4-6'],
    }
    const result = collectPreRegisterModels(settings)
    expect(result).toEqual(['claude-opus-4-6'])
  })

  it('ignores defaultModel when defaultProvider is not vertex-anthropic', () => {
    const settings = {
      defaultProvider: 'koji',
      defaultModel: 'some-model',
      enabledModels: [],
    }
    const result = collectPreRegisterModels(settings)
    expect(result).toEqual([])
  })

  it('returns empty array when no matching models', () => {
    const settings = {
      enabledModels: ['koji/model-a', 'other/model-b'],
    }
    const result = collectPreRegisterModels(settings)
    expect(result).toEqual([])
  })
})
