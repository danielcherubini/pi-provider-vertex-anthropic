import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { readFileSync } from 'node:fs'

vi.mock('node:fs', () => ({
  readFileSync: vi.fn(),
  existsSync: vi.fn(),
}))

// Import after mocking so the module picks up the mock
const { readModelsFromJson } = await import('../src/models-json')

describe('readModelsFromJson', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('returns models when vertex-anthropic models are defined', () => {
    const modelsJson = {
      providers: {
        'vertex-anthropic': {
          models: [
            {
              id: 'claude-sonnet-4-6@default',
              name: 'Claude Sonnet 4.6',
              reasoning: true,
              input: ['text', 'image'],
              contextWindow: 200000,
              maxTokens: 32000,
              cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
            },
          ],
        },
      },
    }
    vi.mocked(readFileSync).mockReturnValue(JSON.stringify(modelsJson))
    expect(readModelsFromJson()).toEqual(modelsJson.providers['vertex-anthropic'].models)
  })

  it('returns null when models.json does not exist', () => {
    vi.mocked(readFileSync).mockImplementation(() => {
      throw new Error('ENOENT: no such file or directory')
    })
    expect(readModelsFromJson()).toBeNull()
  })

  it('returns null when models.json is empty object', () => {
    vi.mocked(readFileSync).mockReturnValue('{}')
    expect(readModelsFromJson()).toBeNull()
  })

  it('returns null when vertex-anthropic provider has no models key', () => {
    const modelsJson = {
      providers: {
        'vertex-anthropic': {},
      },
    }
    vi.mocked(readFileSync).mockReturnValue(JSON.stringify(modelsJson))
    expect(readModelsFromJson()).toBeNull()
  })

  it('returns null when vertex-anthropic models array is empty', () => {
    const modelsJson = {
      providers: {
        'vertex-anthropic': {
          models: [],
        },
      },
    }
    vi.mocked(readFileSync).mockReturnValue(JSON.stringify(modelsJson))
    expect(readModelsFromJson()).toBeNull()
  })

  it('returns null when only other providers are defined', () => {
    const modelsJson = {
      providers: {
        openai: {
          models: [{ id: 'gpt-4o', name: 'GPT-4o' }],
        },
        google: {
          models: [{ id: 'gemini-2.5-pro', name: 'Gemini 2.5 Pro' }],
        },
      },
    }
    vi.mocked(readFileSync).mockReturnValue(JSON.stringify(modelsJson))
    expect(readModelsFromJson()).toBeNull()
  })

  it('only returns vertex-anthropic models when multiple providers exist', () => {
    const vertexModels = [
      {
        id: 'claude-opus-4-7@default',
        name: 'Claude Opus 4.7',
        reasoning: true,
        input: ['text', 'image'],
        contextWindow: 200000,
        maxTokens: 64000,
        cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
      },
    ]
    const modelsJson = {
      providers: {
        openai: {
          models: [{ id: 'gpt-4o', name: 'GPT-4o' }],
        },
        'vertex-anthropic': {
          models: vertexModels,
        },
      },
    }
    vi.mocked(readFileSync).mockReturnValue(JSON.stringify(modelsJson))
    expect(readModelsFromJson()).toEqual(vertexModels)
  })

  it('returns null when models.json contains invalid JSON', () => {
    vi.mocked(readFileSync).mockReturnValue('not valid json {{{')
    expect(readModelsFromJson()).toBeNull()
  })

  it('returns null when models is not an array', () => {
    const modelsJson = {
      providers: {
        'vertex-anthropic': {
          models: 'not-an-array',
        },
      },
    }
    vi.mocked(readFileSync).mockReturnValue(JSON.stringify(modelsJson))
    expect(readModelsFromJson()).toBeNull()
  })
})
