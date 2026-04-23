import type {
  BetaMessage,
  BetaMessageParam,
  BetaRawMessageStreamEvent,
  BetaToolChoice,
  BetaToolUnion,
  BetaUsage,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'

type AnyBlock = Record<string, unknown>

export type OpenAICompatConfig = {
  apiKey: string
  baseURL: string
  headers?: Record<string, string>
  fetch?: typeof globalThis.fetch
}

type OpenAIToolCall = {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

type OpenAIChatContentPart =
  | { type: 'text'; text: string }
  | { type: 'image_url'; image_url: { url: string } }

type OpenAIChatMessage = {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content?: string | OpenAIChatContentPart[] | null
  tool_call_id?: string
  tool_calls?: OpenAIToolCall[]
}

export type OpenAIChatRequest = {
  model: string
  messages: OpenAIChatMessage[]
  stream?: boolean
  enable_thinking?: boolean
  thinking_budget?: number
  temperature?: number
  tools?: Array<{
    type: 'function'
    function: {
      name: string
      description?: string
      parameters?: unknown
    }
  }>
  tool_choice?:
    | 'auto'
    | 'required'
    | 'none'
    | { type: 'function'; function: { name: string } }
  max_tokens?: number
}

type OpenAIStreamChunk = {
  id?: string
  model?: string
  choices?: Array<{
    index?: number
    delta?: {
      role?: 'assistant'
      content?: string | null
      reasoning_content?: string | null
      tool_calls?: Array<{
        index?: number
        id?: string
        type?: 'function'
        function?: {
          name?: string
          arguments?: string
        }
      }>
    }
    finish_reason?: string | null
  }>
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
  }
}

export function joinBaseUrl(baseURL: string, path: string): string {
  return `${baseURL.replace(/\/$/, '')}${path}`
}

export function contentToText(content: BetaMessageParam['content']): string {
  if (typeof content === 'string') return content
  return content
    .map(block => {
      if (block.type === 'text') return typeof block.text === 'string' ? block.text : ''
      if (block.type === 'tool_result') {
        return typeof block.content === 'string'
          ? block.content
          : JSON.stringify(block.content)
      }
      return ''
    })
    .filter(Boolean)
    .join('\n')
}

export function toBlocks(content: BetaMessageParam['content']): AnyBlock[] {
  return Array.isArray(content)
    ? (content as unknown as AnyBlock[])
    : [{ type: 'text', text: content }]
}

function toDataUrl(mediaType: string, data: string): string {
  return `data:${mediaType};base64,${data}`
}

function mapAnthropicUserBlocksToOpenAIContent(
  blocks: AnyBlock[],
): OpenAIChatContentPart[] {
  return blocks.flatMap(block => {
    if (block.type === 'text' && typeof block.text === 'string' && block.text.length > 0) {
      return [{ type: 'text' as const, text: block.text }]
    }
    if (block.type === 'image' && block.source && typeof block.source === 'object') {
      const source = block.source as Record<string, unknown>
      if (
        source.type === 'base64' &&
        typeof source.media_type === 'string' &&
        typeof source.data === 'string'
      ) {
        return [{
          type: 'image_url' as const,
          image_url: { url: toDataUrl(String(source.media_type), String(source.data)) },
        }]
      }
      if (source.type === 'url' && typeof source.url === 'string') {
        return [{ type: 'image_url' as const, image_url: { url: String(source.url) } }]
      }
    }
    if (block.type === 'document' && block.source && typeof block.source === 'object') {
      const source = block.source as Record<string, unknown>
      if (source.type === 'text' && typeof source.data === 'string') {
        return [{ type: 'text' as const, text: String(source.data) }]
      }
    }
    return []
  })
}

function stripSchemaMetaKeys(schema: unknown): unknown {
  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) return schema
  const { $schema: _s, $id: _i, ...rest } = schema as Record<string, unknown>
  return rest
}

export function getToolDefinitions(tools?: BetaToolUnion[]): OpenAIChatRequest['tools'] {
  if (!tools || tools.length === 0) return undefined
  const mapped = tools.flatMap(tool => {
    const record = tool as unknown as Record<string, unknown>
    const name = typeof record.name === 'string' ? record.name : undefined
    if (!name) return []
    return [{
      type: 'function' as const,
      function: {
        name,
        description:
          typeof record.description === 'string' ? record.description : undefined,
        parameters: stripSchemaMetaKeys(record.input_schema),
      },
    }]
  })
  return mapped.length > 0 ? mapped : undefined
}

export function convertAnthropicRequestToOpenAI(input: {
  model: string
  system?: string | Array<{ type?: string; text?: string }>
  messages: BetaMessageParam[]
  tools?: BetaToolUnion[]
  tool_choice?: BetaToolChoice
  temperature?: number
  max_tokens?: number
  thinking?: {
    type?: 'enabled' | 'disabled' | 'adaptive'
    budget_tokens?: number
  }
}): OpenAIChatRequest {
  const configuredModel = process.env.ANTHROPIC_MODEL?.trim()
  const targetModel = configuredModel || input.model
  const messages: OpenAIChatMessage[] = []

  if (input.system) {
    const systemText = Array.isArray(input.system)
      ? input.system.map(block => block.text ?? '').join('\n')
      : input.system
    if (systemText) messages.push({ role: 'system', content: systemText })
  }

  for (const message of input.messages) {
    if (message.role === 'user') {
      const blocks = toBlocks(message.content)

      const toolResults = blocks.filter(block => block.type === 'tool_result')
      const hoistedImages: OpenAIChatContentPart[] = []
      for (const result of toolResults) {
        if (typeof result.tool_use_id !== 'string' || result.tool_use_id.length === 0) {
          throw new Error('[openaiCompat] tool_result missing tool_use_id — cannot correlate with tool_call')
        }
        const toolUseId = result.tool_use_id
        const rawContent = result.content
        let textContent: string
        if (typeof rawContent === 'string') {
          textContent = rawContent
        } else if (Array.isArray(rawContent)) {
          const parts = rawContent as AnyBlock[]
          const imageParts = mapAnthropicUserBlocksToOpenAIContent(
            parts.filter(b => b.type === 'image'),
          )
          if (imageParts.length > 0) hoistedImages.push(...imageParts)
          textContent = parts
            .filter(b => b.type !== 'image')
            .map(b => {
              if (b.type === 'text' && typeof b.text === 'string') return b.text
              return JSON.stringify(b)
            })
            .join('\n')
          if (imageParts.length > 0 && !textContent) {
            textContent = `[tool returned ${imageParts.length} image${imageParts.length === 1 ? '' : 's'}; see next user message]`
          }
        } else {
          textContent = JSON.stringify(rawContent ?? '')
        }
        if (result.is_error === true) textContent = `[tool_error] ${textContent}`
        messages.push({
          role: 'tool',
          tool_call_id: toolUseId,
          content: textContent,
        })
      }

      const userContent = mapAnthropicUserBlocksToOpenAIContent(
        blocks.filter(block => block.type !== 'tool_result') as AnyBlock[],
      )
      const combinedUserContent = [...hoistedImages, ...userContent]
      if (combinedUserContent.length > 0) {
        messages.push({ role: 'user', content: combinedUserContent })
      }
      continue
    }

    if (message.role === 'assistant') {
      const blocks = Array.isArray(message.content)
        ? (message.content as unknown as AnyBlock[])
        : []
      const text = blocks
        .filter(block => block.type === 'text')
        .map(block => (typeof block.text === 'string' ? block.text : ''))
        .join('')

      const toolCalls = blocks
        .filter(block => block.type === 'tool_use')
        .map(block => ({
          id: String(block.id),
          type: 'function' as const,
          function: {
            name: String(block.name),
            arguments:
              typeof block.input === 'string'
                ? block.input
                : JSON.stringify(block.input ?? {}),
          },
        }))

      if (text || toolCalls.length > 0) {
        messages.push({
          role: 'assistant',
          content: text || null,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        })
      }
    }
  }

  const thinkingEnabled =
    input.thinking?.type === 'enabled' || input.thinking?.type === 'adaptive'
  const toolChoiceType = input.tool_choice?.type
  return {
    model: targetModel,
    messages,
    ...(thinkingEnabled ? { enable_thinking: true } : {}),
    ...(input.thinking?.type === 'enabled' &&
    typeof input.thinking.budget_tokens === 'number'
      ? { thinking_budget: input.thinking.budget_tokens }
      : {}),
    temperature: input.temperature,
    max_tokens: input.max_tokens,
    ...(getToolDefinitions(input.tools)
      ? { tools: getToolDefinitions(input.tools) }
      : {}),
    ...(toolChoiceType === 'tool' && input.tool_choice?.type === 'tool'
      ? {
          tool_choice: {
            type: 'function' as const,
            function: { name: input.tool_choice.name },
          },
        }
      : toolChoiceType === 'auto'
        ? { tool_choice: 'auto' as const }
        : toolChoiceType === 'any'
          ? { tool_choice: 'required' as const }
          : toolChoiceType === 'none'
            ? { tool_choice: 'none' as const }
            : {}),
  }
}

export async function createOpenAICompatStream(
  config: OpenAICompatConfig,
  request: OpenAIChatRequest,
  signal?: AbortSignal,
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const response = await (config.fetch ?? globalThis.fetch)(
    joinBaseUrl(config.baseURL, '/chat/completions'),
    {
      method: 'POST',
      signal,
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${config.apiKey}`,
        ...config.headers,
      },
      body: JSON.stringify({
        ...request,
        stream: true,
        stream_options: { include_usage: true },
      }),
    },
  )

  if (!response.ok || !response.body) {
    let responseText = ''
    try {
      responseText = await response.text()
    } catch {
      responseText = ''
    }
    throw new Error(
      `OpenAI compatible request failed with status ${response.status}${responseText ? `: ${responseText}` : ''}`,
    )
  }

  const contentType = response.headers.get('content-type') ?? ''
  if (!contentType.includes('text/event-stream')) {
    let responseText = ''
    try {
      responseText = await response.text()
    } catch {
      responseText = ''
    }
    throw new Error(
      `OpenAI compatible endpoint returned non-streaming response (content-type: ${contentType || 'unknown'}): ${responseText.slice(0, 500)}`,
    )
  }

  return response.body.getReader()
}

export function parseSSEChunk(buffer: string): { events: string[]; remainder: string } {
  const normalized = buffer.replace(/\r\n/g, '\n')
  const parts = normalized.split('\n\n')
  const remainder = parts.pop() ?? ''
  return { events: parts, remainder }
}

export function mapFinishReason(reason: string | null | undefined): BetaMessage['stop_reason'] {
  if (reason === 'tool_calls') return 'tool_use'
  if (reason === 'length') return 'max_tokens'
  return 'end_turn'
}

export async function* createAnthropicStreamFromOpenAI(input: {
  reader: ReadableStreamDefaultReader<Uint8Array>
  model: string
}): AsyncGenerator<BetaRawMessageStreamEvent, BetaMessage, void> {
  const decoder = new TextDecoder()
  let buffer = ''
  let started = false
  let currentTextIndex: number | null = null
  let currentThinkingIndex: number | null = null
  let nextContentIndex = 0
  let promptTokens = 0
  let completionTokens = 0
  let emittedAnyContent = false
  let responseId = 'openai-compat'
  let stopReason: BetaMessage['stop_reason'] = 'end_turn'
  let finishSeen = false
  const toolCallState = new Map<number, { id: string; name: string; arguments: string }>()
  const toolCallOrder: number[] = []

  while (true) {
    const { done, value } = await input.reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const parsed = parseSSEChunk(buffer)
    buffer = parsed.remainder

    for (const rawEvent of parsed.events) {
      const dataLines = rawEvent
        .split('\n')
        .filter(line => line.startsWith('data:'))
        .map(line => line.slice(5).trim())

      for (const data of dataLines) {
        if (!data || data === '[DONE]') continue
        const chunk = JSON.parse(data) as OpenAIStreamChunk & {
          error?: { message?: string; code?: string | number; type?: string }
        }
        if (!chunk || typeof chunk !== 'object') {
          throw new Error(
            `[openaiCompat] invalid stream chunk: ${String(data).slice(0, 500)}`,
          )
        }
        if (chunk.error) {
          const msg =
            chunk.error.message ||
            chunk.error.type ||
            JSON.stringify(chunk.error).slice(0, 500)
          throw new Error(`[openaiCompat] stream error: ${msg}`)
        }

        if (chunk.usage) {
          promptTokens = chunk.usage.prompt_tokens ?? promptTokens
          completionTokens = chunk.usage.completion_tokens ?? completionTokens
        }
        if (chunk.id) responseId = chunk.id

        if (!started) {
          started = true
          yield {
            type: 'message_start',
            message: {
              id: responseId,
              type: 'message',
              role: 'assistant',
              model: input.model,
              content: [],
              stop_reason: null,
              stop_sequence: null,
              usage: {
                input_tokens: promptTokens,
                output_tokens: 0,
              },
            },
          } as BetaRawMessageStreamEvent
        }

        const choice = chunk.choices?.[0]
        if (!choice) continue
        const delta = choice.delta

        if (delta?.reasoning_content) {
          if (currentTextIndex !== null) {
            yield { type: 'content_block_stop', index: currentTextIndex } as BetaRawMessageStreamEvent
            currentTextIndex = null
          }
          if (currentThinkingIndex === null) {
            currentThinkingIndex = nextContentIndex++
            yield {
              type: 'content_block_start',
              index: currentThinkingIndex,
              content_block: { type: 'thinking', thinking: '', signature: '' },
            } as BetaRawMessageStreamEvent
          }
          yield {
            type: 'content_block_delta',
            index: currentThinkingIndex,
            delta: { type: 'thinking_delta', thinking: delta.reasoning_content },
          } as BetaRawMessageStreamEvent
          emittedAnyContent = true
        }

        if (delta?.content) {
          if (currentThinkingIndex !== null) {
            yield { type: 'content_block_stop', index: currentThinkingIndex } as BetaRawMessageStreamEvent
            currentThinkingIndex = null
          }
          if (currentTextIndex === null) {
            currentTextIndex = nextContentIndex++
            yield {
              type: 'content_block_start',
              index: currentTextIndex,
              content_block: { type: 'text', text: '' },
            } as BetaRawMessageStreamEvent
          }
          yield {
            type: 'content_block_delta',
            index: currentTextIndex,
            delta: { type: 'text_delta', text: delta.content },
          } as BetaRawMessageStreamEvent
          emittedAnyContent = true
        }

        for (const toolCall of delta?.tool_calls ?? []) {
          const openAIIndex = toolCall.index ?? 0
          let state = toolCallState.get(openAIIndex)
          if (!state) {
            state = {
              id: toolCall.id ?? `toolu_openai_${openAIIndex}`,
              name: toolCall.function?.name ?? '',
              arguments: '',
            }
            toolCallState.set(openAIIndex, state)
            toolCallOrder.push(openAIIndex)
          }
          if (toolCall.id) state.id = toolCall.id
          if (toolCall.function?.name) state.name = toolCall.function.name
          if (toolCall.function?.arguments) state.arguments += toolCall.function.arguments
        }

        if (choice.finish_reason && !finishSeen) {
          finishSeen = true
          stopReason = mapFinishReason(choice.finish_reason)
        }
      }
    }
  }

  if (!started) {
    throw new Error(
      `[openaiCompat] stream ended before any event for model=${input.model}`,
    )
  }

  if (currentTextIndex !== null) {
    yield { type: 'content_block_stop', index: currentTextIndex } as BetaRawMessageStreamEvent
    currentTextIndex = null
  }
  if (currentThinkingIndex !== null) {
    yield { type: 'content_block_stop', index: currentThinkingIndex } as BetaRawMessageStreamEvent
    currentThinkingIndex = null
  }

  for (const openAIIndex of toolCallOrder) {
    const state = toolCallState.get(openAIIndex)
    if (!state) continue
    if (!state.name) {
      throw new Error(`[openaiCompat] tool_call at index ${openAIIndex} has no name`)
    }
    const idx = nextContentIndex++
    yield {
      type: 'content_block_start',
      index: idx,
      content_block: { type: 'tool_use', id: state.id, name: state.name, input: {} },
    } as BetaRawMessageStreamEvent
    yield {
      type: 'content_block_delta',
      index: idx,
      delta: { type: 'input_json_delta', partial_json: state.arguments || '{}' },
    } as BetaRawMessageStreamEvent
    yield { type: 'content_block_stop', index: idx } as BetaRawMessageStreamEvent
    emittedAnyContent = true
  }

  if (toolCallOrder.length > 0 && stopReason !== 'max_tokens') {
    stopReason = 'tool_use'
  }

  if (!emittedAnyContent) {
    const idx = nextContentIndex++
    yield {
      type: 'content_block_start',
      index: idx,
      content_block: { type: 'text', text: '' },
    } as BetaRawMessageStreamEvent
    yield { type: 'content_block_stop', index: idx } as BetaRawMessageStreamEvent
  }

  yield {
    type: 'message_delta',
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: {
      input_tokens: promptTokens,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
      output_tokens: completionTokens,
    },
  } as BetaRawMessageStreamEvent

  yield { type: 'message_stop' } as BetaRawMessageStreamEvent

  return {
    id: responseId,
    type: 'message',
    role: 'assistant',
    model: input.model,
    content: [],
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: promptTokens,
      output_tokens: completionTokens,
    },
  } as BetaMessage
}

export function mapOpenAIUsageToAnthropic(usage?: {
  prompt_tokens?: number
  completion_tokens?: number
}): BetaUsage | undefined {
  if (!usage) return undefined
  return {
    input_tokens: usage.prompt_tokens ?? 0,
    output_tokens: usage.completion_tokens ?? 0,
    cache_creation_input_tokens: 0,
    cache_read_input_tokens: 0,
  } as BetaUsage
}
