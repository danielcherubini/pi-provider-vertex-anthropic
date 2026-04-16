import type {
  Api,
  AssistantMessage,
  AssistantMessageEventStream,
  Context,
  Model,
  SimpleStreamOptions,
  ThinkingContent,
  ToolCall,
} from '@mariozechner/pi-ai'
import { calculateCost, createAssistantMessageEventStream } from '@mariozechner/pi-ai'

import { getAccessToken } from './auth'
import { buildStreamUrl, resolveConfig } from './config'
import { convertMessages, convertTools, mapStopReason, sanitizeSurrogates } from './messages'

/**
 * Parse Server-Sent Events from a Vertex AI streaming response.
 */
export async function* parseSSE(response: Response): AsyncGenerator<any> {
  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split('\n')
    buffer = lines.pop()!

    let eventType = ''
    let data = ''

    for (const line of lines) {
      if (line.startsWith('event: ')) {
        eventType = line.slice(7).trim()
      } else if (line.startsWith('data: ')) {
        data = line.slice(6).trim()
      } else if (line === '' && data) {
        try {
          const parsed = JSON.parse(data)
          parsed._eventType = eventType
          yield parsed
        } catch {
          // Skip malformed JSON but log for debugging
          console.error(
            `[vertex-anthropic] Failed to parse SSE event: ${data.substring(0, 200)}`,
          )
        }
        eventType = ''
        data = ''
      }
    }
  }
}

const THINKING_BUDGETS: Record<string, number> = {
  minimal: 1024,
  low: 4096,
  medium: 10240,
  high: 20480,
}

/**
 * Build the Anthropic Messages API request body for Vertex AI.
 */
export function buildRequestBody(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): any {
  const body: any = {
    anthropic_version: 'vertex-2023-10-16',
    messages: convertMessages(context.messages, model),
    // Default to 1/3 of model maxTokens to leave headroom for system prompt,
    // tool definitions, and intermediate thinking blocks.
    max_tokens: options?.maxTokens || Math.floor(model.maxTokens / 3),
    stream: true,
  }

  if (context.systemPrompt) {
    body.system = [
      {
        type: 'text',
        text: sanitizeSurrogates(context.systemPrompt),
        cache_control: { type: 'ephemeral' },
      },
    ]
  }

  if (context.tools) {
    body.tools = convertTools(context.tools)
  }

  if (options?.reasoning && model.reasoning) {
    const customBudget =
      options.thinkingBudgets?.[options.reasoning as keyof typeof options.thinkingBudgets]
    body.thinking = {
      type: 'enabled',
      budget_tokens: customBudget ?? THINKING_BUDGETS[options.reasoning] ?? 10240,
    }
  }

  return body
}

/**
 * Stream a response from Vertex AI's Anthropic endpoint.
 * Produces a Pi AssistantMessageEventStream with proper event types.
 */
export function streamVertexAnthropic(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions,
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream()

  ;(async () => {
    try {
    const output: AssistantMessage = {
      role: 'assistant',
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: 'stop',
      timestamp: Date.now(),
    }

    try {
      const config = resolveConfig()
      const project = (model as any).project || config.project
      const region = (model as any).region || config.region

      if (!project) {
        throw new Error(
          'No GCP project configured. Set ANTHROPIC_VERTEX_PROJECT_ID or run /login',
        )
      }

      const token = await getAccessToken()
      const vertexModelId = (model as any).vertexModelId || model.id
      const url = buildStreamUrl(region, project, vertexModelId)
      const body = buildRequestBody(model, context, options)

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal: options?.signal,
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Vertex AI error (${response.status}): ${errorText}`)
      }

      stream.push({ type: 'start', partial: output })

      type Block = (ThinkingContent | { type: 'text'; text: string } | (ToolCall & { partialJson: string })) & {
        index: number
      }
      const blocks = output.content as Block[]

      for await (const event of parseSSE(response)) {
        if (event.type === 'message_start') {
          const usage = event.message?.usage
          if (usage) {
            output.usage.input = usage.input_tokens || 0
            output.usage.output = usage.output_tokens || 0
            output.usage.cacheRead = usage.cache_read_input_tokens || 0
            output.usage.cacheWrite = usage.cache_creation_input_tokens || 0
            output.usage.totalTokens =
              output.usage.input +
              output.usage.output +
              output.usage.cacheRead +
              output.usage.cacheWrite
            calculateCost(model, output.usage)
          }
        } else if (event.type === 'content_block_start') {
          const cb = event.content_block
          if (cb.type === 'text') {
            output.content.push({ type: 'text', text: '', index: event.index } as any)
            stream.push({
              type: 'text_start',
              contentIndex: output.content.length - 1,
              partial: output,
            })
          } else if (cb.type === 'thinking') {
            output.content.push({
              type: 'thinking',
              thinking: '',
              thinkingSignature: '',
              index: event.index,
            } as any)
            stream.push({
              type: 'thinking_start',
              contentIndex: output.content.length - 1,
              partial: output,
            })
          } else if (cb.type === 'tool_use') {
            output.content.push({
              type: 'toolCall',
              id: cb.id,
              name: cb.name,
              arguments: {},
              partialJson: '',
              index: event.index,
            } as any)
            stream.push({
              type: 'toolcall_start',
              contentIndex: output.content.length - 1,
              partial: output,
            })
          }
        } else if (event.type === 'content_block_delta') {
          const index = blocks.findIndex((b) => b.index === event.index)
          const block = blocks[index]
          if (!block) continue

          if (event.delta.type === 'text_delta' && block.type === 'text') {
            block.text += event.delta.text
            stream.push({
              type: 'text_delta',
              contentIndex: index,
              delta: event.delta.text,
              partial: output,
            })
          } else if (event.delta.type === 'thinking_delta' && block.type === 'thinking') {
            block.thinking += event.delta.thinking
            stream.push({
              type: 'thinking_delta',
              contentIndex: index,
              delta: event.delta.thinking,
              partial: output,
            })
          } else if (event.delta.type === 'input_json_delta' && block.type === 'toolCall') {
            ;(block as any).partialJson += event.delta.partial_json
            try {
              block.arguments = JSON.parse((block as any).partialJson)
            } catch {
              // Partial JSON, wait for more
            }
            stream.push({
              type: 'toolcall_delta',
              contentIndex: index,
              delta: event.delta.partial_json,
              partial: output,
            })
          } else if (event.delta.type === 'signature_delta' && block.type === 'thinking') {
            block.thinkingSignature =
              (block.thinkingSignature || '') + (event.delta as any).signature
          }
        } else if (event.type === 'content_block_stop') {
          const index = blocks.findIndex((b) => b.index === event.index)
          const block = blocks[index]
          if (!block) continue

          delete (block as any).index

          if (block.type === 'text') {
            stream.push({
              type: 'text_end',
              contentIndex: index,
              content: block.text,
              partial: output,
            })
          } else if (block.type === 'thinking') {
            stream.push({
              type: 'thinking_end',
              contentIndex: index,
              content: block.thinking,
              partial: output,
            })
          } else if (block.type === 'toolCall') {
            try {
              block.arguments = JSON.parse((block as any).partialJson)
            } catch {
              // Keep whatever we have
            }
            delete (block as any).partialJson
            stream.push({
              type: 'toolcall_end',
              contentIndex: index,
              toolCall: block,
              partial: output,
            })
          }
        } else if (event.type === 'message_delta') {
          if (event.delta?.stop_reason) {
            output.stopReason = mapStopReason(event.delta.stop_reason)
          }
          if (event.usage) {
            output.usage.output = event.usage.output_tokens || output.usage.output
            output.usage.totalTokens =
              output.usage.input +
              output.usage.output +
              output.usage.cacheRead +
              output.usage.cacheWrite
            calculateCost(model, output.usage)
          }
        }
      }

      if (options?.signal?.aborted) {
        throw new Error('Request was aborted')
      }

      // Clean up internal tracking properties
      for (const block of output.content) delete (block as any).index

      stream.push({
        type: 'done',
        reason: output.stopReason as 'stop' | 'length' | 'toolUse',
        message: output,
      })
      stream.end()
    } catch (error) {
      for (const block of output.content) {
        delete (block as any).index
        delete (block as any).partialJson
      }
      output.stopReason = options?.signal?.aborted ? 'aborted' : 'error'
      output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error)
      stream.push({ type: 'error', reason: output.stopReason, error: output })
      stream.end()
    }
  } catch (error) {
    // Catch synchronous errors from IIFE setup (e.g., resolveConfig throws)
    const errorMsg = error instanceof Error ? error.message : JSON.stringify(error)
    console.error(`[vertex-anthropic] Synchronous error in stream: ${errorMsg}`)
  }
  })()

  return stream
}
