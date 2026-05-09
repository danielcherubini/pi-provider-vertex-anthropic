import { describe, it, expect } from 'vitest'
import { buildRequestBody } from '../src/vertex-api'
import type { Model, Api, Context } from '@mariozechner/pi-ai'

const mockModel: Model<Api> = {
  provider: 'vertex-anthropic',
  api: 'vertex-anthropic-api' as Api,
  id: 'claude-opus-4-6',
  name: 'Claude Opus 4.6',
  reasoning: true,
  input: ['text', 'image'],
  contextWindow: 200000,
  maxTokens: 64000,
  cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
} as Model<Api>

const nonReasoningModel: Model<Api> = {
  ...mockModel,
  id: 'claude-3-5-sonnet-v2@20241022',
  name: 'Claude 3.5 Sonnet v2',
  reasoning: false,
  maxTokens: 8192,
} as Model<Api>

const baseContext: Context = {
  systemPrompt: 'You are a helpful assistant.',
  messages: [
    { role: 'user', content: 'hello', timestamp: Date.now() } as any,
  ],
}

describe('buildRequestBody', () => {
  it('sets max_tokens from options.maxTokens', () => {
    const body = buildRequestBody(mockModel, baseContext, { maxTokens: 16000 })
    expect(body.max_tokens).toBe(16000)
  })

  it('defaults max_tokens to 1/3 of model.maxTokens', () => {
    const body = buildRequestBody(mockModel, baseContext)
    expect(body.max_tokens).toBe(Math.floor(64000 / 3))
  })

  it('does not add thinking when reasoning is not set', () => {
    const body = buildRequestBody(mockModel, baseContext, { maxTokens: 16000 })
    expect(body.thinking).toBeUndefined()
  })

  it('does not add thinking for non-reasoning models', () => {
    const body = buildRequestBody(nonReasoningModel, baseContext, {
      maxTokens: 4000,
      reasoning: 'medium',
    })
    expect(body.thinking).toBeUndefined()
  })

  it('adds thinking with correct budget for each level', () => {
    for (const [level, expectedBudget] of [
      ['minimal', 1024],
      ['low', 4096],
      ['medium', 10240],
      ['high', 20480],
    ] as const) {
      const body = buildRequestBody(mockModel, baseContext, {
        maxTokens: 32000,
        reasoning: level,
      })
      expect(body.thinking).toEqual({ type: 'enabled', budget_tokens: expectedBudget })
    }
  })

  it('ensures max_tokens > budget_tokens when maxTokens is small', () => {
    // This is the compaction scenario: maxTokens ~13k but budget_tokens ~20k
    const body = buildRequestBody(mockModel, baseContext, {
      maxTokens: 13107,
      reasoning: 'high',
    })
    expect(body.thinking.budget_tokens).toBe(20480)
    expect(body.max_tokens).toBeGreaterThan(body.thinking.budget_tokens)
    expect(body.max_tokens).toBeLessThanOrEqual(mockModel.maxTokens)
  })

  it('ensures max_tokens > budget_tokens for all thinking levels with small maxTokens', () => {
    for (const level of ['minimal', 'low', 'medium', 'high'] as const) {
      const body = buildRequestBody(mockModel, baseContext, {
        maxTokens: 512,
        reasoning: level,
      })
      expect(body.max_tokens).toBeGreaterThan(body.thinking.budget_tokens)
    }
  })

  it('does not inflate max_tokens when it already exceeds budget_tokens', () => {
    const body = buildRequestBody(mockModel, baseContext, {
      maxTokens: 32000,
      reasoning: 'medium', // budget = 10240
    })
    // max_tokens should stay at 32000 since 32000 > 10240
    expect(body.max_tokens).toBe(32000)
    expect(body.thinking.budget_tokens).toBe(10240)
  })

  it('caps adjusted max_tokens at model.maxTokens and clamps budget', () => {
    const smallModel: Model<Api> = {
      ...mockModel,
      maxTokens: 8192,
    } as Model<Api>

    const body = buildRequestBody(smallModel, baseContext, {
      maxTokens: 4000,
      reasoning: 'high', // budget = 20480, way over model max
    })
    // max_tokens capped to model limit, budget clamped below it
    expect(body.max_tokens).toBeLessThanOrEqual(smallModel.maxTokens)
    expect(body.max_tokens).toBeGreaterThan(body.thinking.budget_tokens)
    // budget should be model.maxTokens - minOutputTokens (1024)
    expect(body.thinking.budget_tokens).toBe(8192 - 1024)
  })

  it('uses custom thinkingBudgets when provided', () => {
    const body = buildRequestBody(mockModel, baseContext, {
      maxTokens: 32000,
      reasoning: 'medium',
      thinkingBudgets: { medium: 5000 },
    })
    expect(body.thinking.budget_tokens).toBe(5000)
  })
})
