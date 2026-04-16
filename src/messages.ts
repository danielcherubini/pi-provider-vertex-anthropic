import type {
  AssistantMessage,
  ImageContent,
  Message,
  Model,
  Api,
  StopReason,
  TextContent,
  ThinkingContent,
  Tool,
  ToolCall,
  ToolResultMessage,
} from '@mariozechner/pi-ai'

/**
 * Replace unpaired surrogate code units that would break JSON serialization.
 */
export function sanitizeSurrogates(text: string): string {
  return text.replace(/[\uD800-\uDFFF]/g, '\uFFFD')
}

/**
 * Normalize a tool call ID to match Anthropic's required pattern:
 * alphanumeric, underscores, and hyphens only, max 64 characters.
 */
export function normalizeToolCallId(id: string): string {
  return id.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 64)
}

/**
 * Map Anthropic API stop reasons to Pi's StopReason type.
 */
export function mapStopReason(reason: string): StopReason {
  switch (reason) {
    case 'end_turn':
    case 'pause_turn':
    case 'stop_sequence':
      return 'stop'
    case 'max_tokens':
      return 'length'
    case 'tool_use':
      return 'toolUse'
    default:
      return 'error'
  }
}

/**
 * Convert a mixed array of text/image content blocks to the Anthropic API format.
 * Returns a plain string when there are no images (optimisation for text-only).
 */
function convertContentBlocks(
  content: (TextContent | ImageContent)[],
): string | Array<{ type: 'text'; text: string } | { type: 'image'; source: any }> {
  const hasImages = content.some((c) => c.type === 'image')

  if (!hasImages) {
    return sanitizeSurrogates(content.map((c) => (c as TextContent).text).join('\n'))
  }

  const blocks = content.map((block) => {
    if (block.type === 'text') {
      return { type: 'text' as const, text: sanitizeSurrogates(block.text) }
    }
    return {
      type: 'image' as const,
      source: {
        type: 'base64' as const,
        media_type: (block as ImageContent).mimeType,
        data: (block as ImageContent).data,
      },
    }
  })

  // Ensure there's at least one text block for the API
  if (!blocks.some((b) => b.type === 'text')) {
    blocks.unshift({ type: 'text' as const, text: '(see attached image)' })
  }

  return blocks
}

/**
 * Transform messages for cross-provider compatibility.
 *
 * - Removes errored/aborted assistant messages
 * - Inserts synthetic tool results for orphaned tool calls
 * - Normalizes tool call IDs for Anthropic's format
 * - Handles thinking blocks (preserves signatures for same-model replay)
 */
export function transformMessages(
  messages: Message[],
  model: Model<Api>,
  idNormalizer?: (id: string) => string,
): Message[] {
  const toolCallIdMap = new Map<string, string>()

  // First pass: normalize content blocks and tool call IDs
  const transformed = messages.map((msg) => {
    if (msg.role === 'user') return msg

    if (msg.role === 'toolResult') {
      const normalizedId = toolCallIdMap.get(msg.toolCallId)
      if (normalizedId && normalizedId !== msg.toolCallId) {
        return { ...msg, toolCallId: normalizedId }
      }
      return msg
    }

    if (msg.role === 'assistant') {
      const assistantMsg = msg as AssistantMessage
      const isSameModel =
        assistantMsg.provider === model.provider &&
        assistantMsg.api === model.api &&
        assistantMsg.model === model.id

      const transformedContent = assistantMsg.content.flatMap((block) => {
        if (block.type === 'thinking') {
          const thinkingBlock = block as ThinkingContent
          if (isSameModel && thinkingBlock.thinkingSignature) return block
          if (!thinkingBlock.thinking || thinkingBlock.thinking.trim() === '') return []
          if (isSameModel) return block
          return { type: 'text' as const, text: thinkingBlock.thinking }
        }

        if (block.type === 'text') {
          if (isSameModel) return block
          return { type: 'text' as const, text: block.text }
        }

        if (block.type === 'toolCall') {
          const toolCall = block as ToolCall
          if (!isSameModel && idNormalizer) {
            const normalizedId = idNormalizer(toolCall.id)
            if (normalizedId !== toolCall.id) {
              toolCallIdMap.set(toolCall.id, normalizedId)
              return { ...toolCall, id: normalizedId }
            }
          }
          return toolCall
        }

        return block
      })

      return { ...assistantMsg, content: transformedContent }
    }

    return msg
  })

  // Second pass: insert synthetic tool results for orphaned tool calls
  const result: Message[] = []
  let pendingToolCalls: ToolCall[] = []
  let seenToolResultIds = new Set<string>()

  function flushOrphaned() {
    for (const tc of pendingToolCalls) {
      if (!seenToolResultIds.has(tc.id)) {
        result.push({
          role: 'toolResult',
          toolCallId: tc.id,
          toolName: tc.name,
          content: [{ type: 'text', text: 'No result provided' }],
          isError: true,
          timestamp: Date.now(),
        } as ToolResultMessage)
      }
    }
    pendingToolCalls = []
    seenToolResultIds = new Set()
  }

  for (const msg of transformed) {
    if (msg.role === 'assistant') {
      if (pendingToolCalls.length > 0) flushOrphaned()

      const assistantMsg = msg as AssistantMessage
      if (assistantMsg.stopReason === 'error' || assistantMsg.stopReason === 'aborted') {
        continue
      }

      const toolCalls = assistantMsg.content.filter((b) => b.type === 'toolCall') as ToolCall[]
      if (toolCalls.length > 0) {
        pendingToolCalls = toolCalls
        seenToolResultIds = new Set()
      }

      result.push(msg)
    } else if (msg.role === 'toolResult') {
      seenToolResultIds.add((msg as ToolResultMessage).toolCallId)
      result.push(msg)
    } else if (msg.role === 'user') {
      if (pendingToolCalls.length > 0) flushOrphaned()
      result.push(msg)
    } else {
      result.push(msg)
    }
  }

  return result
}

/**
 * Convert Pi messages to the Anthropic Messages API format.
 * Handles user, assistant, and toolResult messages with proper batching.
 * Adds ephemeral cache control to the last user message.
 */
export function convertMessages(messages: Message[], model: Model<Api>): any[] {
  const params: any[] = []
  const transformedMessages = transformMessages(messages, model, normalizeToolCallId)

  for (let i = 0; i < transformedMessages.length; i++) {
    const msg = transformedMessages[i]

    if (msg.role === 'user') {
      if (typeof msg.content === 'string') {
        if (msg.content.trim()) {
          params.push({ role: 'user', content: sanitizeSurrogates(msg.content) })
        }
      } else {
        const blocks: any[] = msg.content.map((item) =>
          item.type === 'text'
            ? { type: 'text' as const, text: sanitizeSurrogates(item.text) }
            : {
                type: 'image' as const,
                source: {
                  type: 'base64' as const,
                  media_type: (item as ImageContent).mimeType as any,
                  data: (item as ImageContent).data,
                },
              },
        )
        if (blocks.length > 0) {
          params.push({ role: 'user', content: blocks })
        }
      }
    } else if (msg.role === 'assistant') {
      const blocks: any[] = []
      for (const block of msg.content) {
        if (block.type === 'text' && block.text.trim()) {
          blocks.push({ type: 'text', text: sanitizeSurrogates(block.text) })
        } else if (block.type === 'thinking' && block.thinking.trim()) {
          if ((block as ThinkingContent).thinkingSignature) {
            blocks.push({
              type: 'thinking' as any,
              thinking: sanitizeSurrogates(block.thinking),
              signature: (block as ThinkingContent).thinkingSignature!,
            })
          } else {
            blocks.push({ type: 'text', text: sanitizeSurrogates(block.thinking) })
          }
        } else if (block.type === 'toolCall') {
          blocks.push({
            type: 'tool_use',
            id: block.id,
            name: block.name,
            input: block.arguments,
          })
        }
      }
      if (blocks.length > 0) {
        params.push({ role: 'assistant', content: blocks })
      }
    } else if (msg.role === 'toolResult') {
      const toolResults: any[] = []
      toolResults.push({
        type: 'tool_result',
        tool_use_id: msg.toolCallId,
        content: convertContentBlocks(msg.content),
        is_error: msg.isError,
      })

      // Batch consecutive tool results into a single user message
      let j = i + 1
      while (j < transformedMessages.length && transformedMessages[j].role === 'toolResult') {
        const nextMsg = transformedMessages[j] as ToolResultMessage
        toolResults.push({
          type: 'tool_result',
          tool_use_id: nextMsg.toolCallId,
          content: convertContentBlocks(nextMsg.content),
          is_error: nextMsg.isError,
        })
        j++
      }
      i = j - 1
      params.push({ role: 'user', content: toolResults })
    }
  }

  // Add ephemeral cache control to the last block of the last user message.
  // This enables prompt caching for the most recent context, reducing latency
  // and cost on repeated calls with similar prompts.
  if (params.length > 0) {
    const last = params[params.length - 1]
    if (last.role === 'user' && Array.isArray(last.content)) {
      const lastBlock = last.content[last.content.length - 1]
      if (lastBlock) {
        lastBlock.cache_control = { type: 'ephemeral' }
      }
    }
  }

  return params
}

/**
 * Convert Pi tool definitions to the Anthropic tool format.
 */
export function convertTools(tools: Tool[]): any[] {
  return tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    input_schema: {
      type: 'object',
      properties: (tool.parameters as any).properties || {},
      required: (tool.parameters as any).required || [],
    },
  }))
}
