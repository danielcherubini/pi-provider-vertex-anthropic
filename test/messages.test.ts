import { describe, it, expect } from 'vitest'
import {
  sanitizeSurrogates,
  normalizeToolCallId,
  mapStopReason,
  convertTools,
  transformMessages,
} from '../src/messages'
import type { Message, AssistantMessage, ToolResultMessage, Tool, Model, Api } from '@mariozechner/pi-ai'

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

describe('sanitizeSurrogates', () => {
  it('replaces lone surrogates with replacement character', () => {
    expect(sanitizeSurrogates('hello\uD800world')).toBe('hello\uFFFDworld')
  })

  it('leaves normal text unchanged', () => {
    expect(sanitizeSurrogates('hello world')).toBe('hello world')
  })

  it('handles strings with no surrogates', () => {
    expect(sanitizeSurrogates('abc 123 !@#')).toBe('abc 123 !@#')
  })
})

describe('normalizeToolCallId', () => {
  it('replaces non-alphanumeric characters with underscores', () => {
    expect(normalizeToolCallId('tool.call:123')).toBe('tool_call_123')
  })

  it('preserves hyphens and underscores', () => {
    expect(normalizeToolCallId('tool-call_123')).toBe('tool-call_123')
  })

  it('truncates to 64 characters', () => {
    const longId = 'a'.repeat(100)
    expect(normalizeToolCallId(longId)).toHaveLength(64)
  })

  it('leaves valid IDs unchanged', () => {
    expect(normalizeToolCallId('toolu_abc123')).toBe('toolu_abc123')
  })
})

describe('mapStopReason', () => {
  it('maps end_turn to stop', () => {
    expect(mapStopReason('end_turn')).toBe('stop')
  })

  it('maps pause_turn to stop', () => {
    expect(mapStopReason('pause_turn')).toBe('stop')
  })

  it('maps stop_sequence to stop', () => {
    expect(mapStopReason('stop_sequence')).toBe('stop')
  })

  it('maps max_tokens to length', () => {
    expect(mapStopReason('max_tokens')).toBe('length')
  })

  it('maps tool_use to toolUse', () => {
    expect(mapStopReason('tool_use')).toBe('toolUse')
  })

  it('maps unknown reasons to error', () => {
    expect(mapStopReason('unknown')).toBe('error')
  })
})

describe('convertTools', () => {
  it('converts Pi tools to Anthropic format', () => {
    const tools: Tool[] = [
      {
        name: 'search',
        description: 'Search the web',
        parameters: {
          type: 'object',
          properties: { query: { type: 'string' } },
          required: ['query'],
        },
      },
    ]

    const result = convertTools(tools)
    expect(result).toEqual([
      {
        name: 'search',
        description: 'Search the web',
        input_schema: {
          type: 'object',
          properties: { query: { type: 'string' } },
          required: ['query'],
        },
      },
    ])
  })

  it('defaults to empty properties and required', () => {
    const tools: Tool[] = [
      { name: 'noop', description: 'Does nothing', parameters: {} as any },
    ]

    const result = convertTools(tools)
    expect(result[0].input_schema.properties).toEqual({})
    expect(result[0].input_schema.required).toEqual([])
  })
})

describe('transformMessages', () => {
  it('passes user messages through unchanged', () => {
    const messages: Message[] = [
      { role: 'user', content: 'hello', timestamp: Date.now() } as any,
    ]
    const result = transformMessages(messages, mockModel)
    expect(result).toEqual(messages)
  })

  it('removes errored assistant messages', () => {
    const messages: Message[] = [
      { role: 'user', content: 'hello', timestamp: Date.now() } as any,
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'partial response' }],
        stopReason: 'error',
        provider: 'vertex-anthropic',
        api: 'vertex-anthropic-api',
        model: 'claude-opus-4-6',
        timestamp: Date.now(),
      } as AssistantMessage,
    ]
    const result = transformMessages(messages, mockModel)
    expect(result).toHaveLength(1)
    expect(result[0].role).toBe('user')
  })

  it('removes aborted assistant messages', () => {
    const messages: Message[] = [
      { role: 'user', content: 'hello', timestamp: Date.now() } as any,
      {
        role: 'assistant',
        content: [{ type: 'text', text: 'aborted' }],
        stopReason: 'aborted',
        provider: 'vertex-anthropic',
        api: 'vertex-anthropic-api',
        model: 'claude-opus-4-6',
        timestamp: Date.now(),
      } as AssistantMessage,
    ]
    const result = transformMessages(messages, mockModel)
    expect(result).toHaveLength(1)
  })

  it('inserts synthetic tool results for orphaned tool calls', () => {
    const messages: Message[] = [
      { role: 'user', content: 'hello', timestamp: Date.now() } as any,
      {
        role: 'assistant',
        content: [
          { type: 'toolCall', id: 'tc_1', name: 'search', arguments: { q: 'test' } },
        ],
        stopReason: 'toolUse',
        provider: 'vertex-anthropic',
        api: 'vertex-anthropic-api',
        model: 'claude-opus-4-6',
        timestamp: Date.now(),
      } as AssistantMessage,
      { role: 'user', content: 'nevermind', timestamp: Date.now() } as any,
    ]
    const result = transformMessages(messages, mockModel)

    // Should have: user, assistant, synthetic toolResult, user
    expect(result).toHaveLength(4)
    expect(result[2].role).toBe('toolResult')
    expect((result[2] as ToolResultMessage).toolCallId).toBe('tc_1')
    expect((result[2] as ToolResultMessage).isError).toBe(true)
  })

  it('does not insert synthetic results when tool results exist', () => {
    const messages: Message[] = [
      { role: 'user', content: 'hello', timestamp: Date.now() } as any,
      {
        role: 'assistant',
        content: [
          { type: 'toolCall', id: 'tc_1', name: 'search', arguments: { q: 'test' } },
        ],
        stopReason: 'toolUse',
        provider: 'vertex-anthropic',
        api: 'vertex-anthropic-api',
        model: 'claude-opus-4-6',
        timestamp: Date.now(),
      } as AssistantMessage,
      {
        role: 'toolResult',
        toolCallId: 'tc_1',
        toolName: 'search',
        content: [{ type: 'text', text: 'result' }],
        isError: false,
        timestamp: Date.now(),
      } as ToolResultMessage,
    ]
    const result = transformMessages(messages, mockModel)
    expect(result).toHaveLength(3)
  })

  it('converts thinking blocks to text for different models', () => {
    const otherModel: Model<Api> = {
      ...mockModel,
      provider: 'other-provider',
    } as Model<Api>

    const messages: Message[] = [
      { role: 'user', content: 'hello', timestamp: Date.now() } as any,
      {
        role: 'assistant',
        content: [
          { type: 'thinking', thinking: 'Let me think...', thinkingSignature: '' },
          { type: 'text', text: 'Hello!' },
        ],
        stopReason: 'stop',
        provider: 'other-provider',
        api: 'other-api',
        model: 'other-model',
        timestamp: Date.now(),
      } as AssistantMessage,
    ]

    const result = transformMessages(messages, otherModel)
    const assistant = result[1] as AssistantMessage
    // Thinking without signature from different model should be converted to text
    expect(assistant.content[0].type).toBe('text')
  })
})
