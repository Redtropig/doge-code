import type {
  BetaMessage,
  BetaMessageParam,
  BetaRawMessageStreamEvent,
  BetaToolChoice,
  BetaToolUnion,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import type { EffortValue } from 'src/utils/effort.js'
import {
  getToolDefinitions,
  joinBaseUrl,
  parseSSEChunk,
  toBlocks,
  type OpenAICompatConfig,
} from './openaiCompat.js'

type OpenAIResponsesInputPart =
  | { type: 'input_text'; text: string }
  | { type: 'input_image'; image_url: string }
  | { type: 'output_text'; text: string }

type OpenAIResponsesReasoningSummaryPart = {
  type: 'summary_text'
  text: string
}

type OpenAIResponsesInputItem = {
  type: string
  role?: 'system' | 'user' | 'assistant'
  content?: OpenAIResponsesInputPart[]
  call_id?: string
  name?: string
  arguments?: string
  output?: string
  id?: string
  summary?: OpenAIResponsesReasoningSummaryPart[]
  encrypted_content?: string
}

type OpenAIResponsesRequest = {
  model: string
  input: OpenAIResponsesInputItem[]
  stream?: boolean
  temperature?: number
  max_output_tokens?: number
  tools?: Array<{
    type: 'function'
    name: string
    description?: string
    parameters?: unknown
  }>
  tool_choice?: 'auto' | 'required' | 'none' | { type: 'function'; name: string }
  reasoning?: {
    effort?: 'low' | 'medium' | 'high'
    summary?: 'auto'
  }
  include?: string[]
}

type OpenAIResponsesEvent = {
  type?: string
  response_id?: string
  item_id?: string
  output_index?: number
  item?: {
    type?: string
    id?: string
    call_id?: string
    name?: string
    arguments?: string
    encrypted_content?: string
    [k: string]: unknown
  }
  delta?: string
  arguments_delta?: string
  summary?: Array<{ text?: string }>
  part?: {
    type?: string
    text?: string
    summary?: Array<{ text?: string }>
  }
  response?: {
    id?: string
    status?: string
    error?: { message?: string; code?: string; type?: string } | null
    incomplete_details?: { reason?: string } | null
    usage?: {
      input_tokens?: number
      output_tokens?: number
    }
  }
  error?: { message?: string; code?: string | number; type?: string }
}

type ToolState = {
  id: string
  name: string
  arguments: string
}

function mapEffortToResponsesReasoning(
  effort?: EffortValue,
): OpenAIResponsesRequest['reasoning'] | undefined {
  if (effort === 'none') return undefined
  if (effort === 'low' || effort === 'medium' || effort === 'high') {
    return {
      effort,
      summary: 'auto',
    }
  }
  if (effort === 'max' || typeof effort === 'number') {
    return {
      effort: 'high',
      summary: 'auto',
    }
  }
  return {
    effort: 'medium',
    summary: 'auto',
  }
}

function toDataUrl(mediaType: string, data: string): string {
  return `data:${mediaType};base64,${data}`
}

function mapAnthropicUserBlocksToResponsesContent(
  blocks: Array<Record<string, unknown>>,
): OpenAIResponsesInputPart[] {
  return blocks.flatMap(block => {
    if (block.type === 'text' && typeof block.text === 'string' && block.text.length > 0) {
      return [{ type: 'input_text' as const, text: block.text }]
    }
    if (block.type === 'image' && block.source && typeof block.source === 'object') {
      const source = block.source as Record<string, unknown>
      if (
        source.type === 'base64' &&
        typeof source.media_type === 'string' &&
        typeof source.data === 'string'
      ) {
        return [{
          type: 'input_image' as const,
          image_url: toDataUrl(String(source.media_type), String(source.data)),
        }]
      }
      if (source.type === 'url' && typeof source.url === 'string') {
        return [{ type: 'input_image' as const, image_url: String(source.url) }]
      }
    }
    if (block.type === 'document' && block.source && typeof block.source === 'object') {
      const source = block.source as Record<string, unknown>
      if (source.type === 'text' && typeof source.data === 'string') {
        return [{ type: 'input_text' as const, text: String(source.data) }]
      }
    }
    return []
  })
}

function getResponsesToolDefinitions(tools?: BetaToolUnion[]): OpenAIResponsesRequest['tools'] {
  const definitions = getToolDefinitions(tools)
  if (!definitions) return undefined
  return definitions.map(tool => ({
    type: 'function' as const,
    name: tool.function.name,
    description: tool.function.description,
    parameters: tool.function.parameters,
  }))
}

export function convertAnthropicRequestToOpenAIResponses(input: {
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
  effort?: EffortValue
}): OpenAIResponsesRequest {
  const configuredModel = process.env.ANTHROPIC_MODEL?.trim()
  const targetModel = configuredModel || input.model
  const items: OpenAIResponsesInputItem[] = []

  if (input.system) {
    const systemText = Array.isArray(input.system)
      ? input.system.map(block => block.text ?? '').join('\n')
      : input.system
    if (systemText) {
      items.push({
        type: 'message',
        role: 'system',
        content: [{ type: 'input_text', text: systemText }],
      })
    }
  }

  for (const message of input.messages) {
    const blocks = toBlocks(message.content)

    if (message.role === 'user') {
      const toolResults = blocks.filter(block => block.type === 'tool_result')
      const hoistedImages: OpenAIResponsesInputPart[] = []
      for (const result of toolResults) {
        if (typeof result.tool_use_id !== 'string' || result.tool_use_id.length === 0) {
          throw new Error('[openaiResponsesCompat] tool_result missing tool_use_id — cannot correlate with function_call')
        }
        const toolUseId = result.tool_use_id
        const rawContent = result.content
        let output: string
        if (typeof rawContent === 'string') {
          output = rawContent
        } else if (Array.isArray(rawContent)) {
          const parts = rawContent as Array<Record<string, unknown>>
          const imageParts = mapAnthropicUserBlocksToResponsesContent(
            parts.filter(b => b.type === 'image'),
          )
          if (imageParts.length > 0) hoistedImages.push(...imageParts)
          output = parts
            .filter(b => b.type !== 'image')
            .map(b => {
              if (b.type === 'text' && typeof b.text === 'string') return b.text
              return JSON.stringify(b)
            })
            .join('\n')
          if (imageParts.length > 0 && !output) {
            output = `[tool returned ${imageParts.length} image${imageParts.length === 1 ? '' : 's'}; see next user message]`
          }
        } else {
          output = JSON.stringify(rawContent ?? '')
        }
        if (result.is_error === true) output = `[tool_error] ${output}`
        items.push({
          type: 'function_call_output',
          call_id: toolUseId,
          output,
        })
      }

      const userContent = mapAnthropicUserBlocksToResponsesContent(
        blocks.filter(block => block.type !== 'tool_result') as Array<Record<string, unknown>>,
      )
      const combinedUserContent = [...hoistedImages, ...userContent]
      if (combinedUserContent.length > 0) {
        items.push({
          type: 'message',
          role: 'user',
          content: combinedUserContent,
        })
      }
      continue
    }

    for (const block of blocks) {
      if (block.type !== 'thinking') continue
      const signature = typeof block.signature === 'string' ? block.signature : ''
      if (!signature) continue
      let parsed: { id?: unknown; encrypted_content?: unknown } | undefined
      try {
        parsed = JSON.parse(signature)
      } catch {
        continue
      }
      if (
        !parsed ||
        typeof parsed.id !== 'string' ||
        typeof parsed.encrypted_content !== 'string'
      ) {
        continue
      }
      const summaryText = typeof block.thinking === 'string' ? block.thinking : ''
      items.push({
        type: 'reasoning',
        id: parsed.id,
        summary: summaryText
          ? [{ type: 'summary_text', text: summaryText }]
          : [],
        encrypted_content: parsed.encrypted_content,
      })
    }

    const text = blocks
      .filter(block => block.type === 'text')
      .map(block => (typeof block.text === 'string' ? block.text : ''))
      .join('')

    if (text) {
      items.push({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text }],
      })
    }

    const toolCalls = blocks.filter(block => block.type === 'tool_use')
    for (const toolCall of toolCalls) {
      items.push({
        type: 'function_call',
        call_id: String(toolCall.id),
        name: String(toolCall.name),
        arguments:
          typeof toolCall.input === 'string'
            ? toolCall.input
            : JSON.stringify(toolCall.input ?? {}),
      })
    }
  }

  return {
    model: targetModel,
    input: items,
    temperature: input.temperature,
    max_output_tokens: input.max_tokens,
    ...(getResponsesToolDefinitions(input.tools)
      ? { tools: getResponsesToolDefinitions(input.tools) }
      : {}),
    ...(input.tool_choice?.type === 'tool'
      ? {
          tool_choice: {
            type: 'function' as const,
            name: input.tool_choice.name,
          },
        }
      : input.tool_choice?.type === 'auto'
        ? { tool_choice: 'auto' as const }
        : input.tool_choice?.type === 'any'
          ? { tool_choice: 'required' as const }
          : input.tool_choice?.type === 'none'
            ? { tool_choice: 'none' as const }
            : {}),
    ...(input.thinking?.type === 'enabled' || input.thinking?.type === 'adaptive'
      ? {
          reasoning:
            mapEffortToResponsesReasoning(input.effort) ??
            { effort: 'medium' as const, summary: 'auto' as const },
          include: ['reasoning.encrypted_content'],
        }
      : {}),
  }
}

export async function createOpenAIResponsesStream(
  config: OpenAICompatConfig,
  request: OpenAIResponsesRequest,
  signal?: AbortSignal,
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const response = await (config.fetch ?? globalThis.fetch)(
    joinBaseUrl(config.baseURL, '/responses'),
    {
      method: 'POST',
      signal,
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${config.apiKey}`,
        ...config.headers,
      },
      body: JSON.stringify({ ...request, stream: true }),
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
      `OpenAI Responses compatible request failed with status ${response.status}${responseText ? `: ${responseText}` : ''}`,
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
      `OpenAI Responses endpoint returned non-streaming response (content-type: ${contentType || 'unknown'}): ${responseText.slice(0, 500)}`,
    )
  }

  return response.body.getReader()
}

function getEventTextDelta(event: OpenAIResponsesEvent): string | undefined {
  if (typeof event.delta === 'string' && event.delta.length > 0) return event.delta
  if (typeof event.part?.text === 'string' && event.part.text.length > 0) return event.part.text
  return undefined
}

function getEventThinkingDelta(event: OpenAIResponsesEvent): string | undefined {
  const eventSummary = Array.isArray(event.summary)
    ? event.summary
        .map(part => (typeof part?.text === 'string' ? part.text : ''))
        .join('')
    : ''
  if (eventSummary.length > 0) return eventSummary

  const partSummary = Array.isArray(event.part?.summary)
    ? event.part.summary
        .map(part => (typeof part?.text === 'string' ? part.text : ''))
        .join('')
    : ''
  if (partSummary.length > 0) return partSummary

  if (
    typeof event.part?.text === 'string' &&
    event.part.text.length > 0 &&
    event.type?.includes('reasoning')
  ) {
    return event.part.text
  }

  if (typeof event.delta === 'string' && event.delta.length > 0 && event.type?.includes('reasoning')) {
    return event.delta
  }

  return undefined
}

function getToolCallDetails(event: OpenAIResponsesEvent): { id?: string; name?: string } {
  const item = event.item ?? {}
  return {
    id:
      typeof item.call_id === 'string'
        ? item.call_id
        : typeof event.item_id === 'string'
          ? event.item_id
          : undefined,
    name: typeof item.name === 'string' ? item.name : undefined,
  }
}

export async function* createAnthropicStreamFromOpenAIResponses(input: {
  reader: ReadableStreamDefaultReader<Uint8Array>
  model: string
}): AsyncGenerator<BetaRawMessageStreamEvent, BetaMessage, void> {
  const decoder = new TextDecoder()
  let buffer = ''
  let started = false
  let currentTextIndex: number | null = null
  let currentThinkingIndex: number | null = null
  let nextContentIndex = 0
  let emittedAnyContent = false
  let promptTokens = 0
  let completionTokens = 0
  let responseId = 'openai-responses-compat'
  let stopReason: BetaMessage['stop_reason'] = 'end_turn'
  const toolStateById = new Map<string, ToolState>()
  const toolIdsInOrder: string[] = []
  let currentReasoningItemId: string | undefined
  let currentReasoningItemEncrypted: string | undefined

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
        const event = JSON.parse(data) as OpenAIResponsesEvent
        if (!event || typeof event !== 'object') {
          throw new Error(
            `[openaiResponsesCompat] invalid stream event: ${String(data).slice(0, 500)}`,
          )
        }
        if (event.error) {
          const msg =
            event.error.message ||
            event.error.type ||
            JSON.stringify(event.error).slice(0, 500)
          throw new Error(`[openaiResponsesCompat] stream error: ${msg}`)
        }

        responseId = event.response?.id ?? event.response_id ?? responseId
        if (event.response?.usage) {
          promptTokens = event.response.usage.input_tokens ?? promptTokens
          completionTokens = event.response.usage.output_tokens ?? completionTokens
        }

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

        const eventType = event.type ?? ''
        const thinkingDelta = getEventThinkingDelta(event)
        const textDelta = getEventTextDelta(event)

        if (eventType === 'response.output_item.added') {
          const addedItem = event.item
          if (addedItem?.type === 'reasoning' && typeof addedItem.id === 'string') {
            currentReasoningItemId = addedItem.id
            currentReasoningItemEncrypted = undefined
          }
        }

        if (eventType === 'response.output_item.done') {
          const doneItem = event.item
          if (doneItem?.type === 'reasoning') {
            if (typeof doneItem.encrypted_content === 'string') {
              currentReasoningItemEncrypted = doneItem.encrypted_content
            }
            if (currentReasoningItemId && currentReasoningItemEncrypted) {
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
                emittedAnyContent = true
              }
              const signature = JSON.stringify({
                id: currentReasoningItemId,
                encrypted_content: currentReasoningItemEncrypted,
              })
              yield {
                type: 'content_block_delta',
                index: currentThinkingIndex,
                delta: { type: 'signature_delta', signature },
              } as BetaRawMessageStreamEvent
              yield { type: 'content_block_stop', index: currentThinkingIndex } as BetaRawMessageStreamEvent
              currentThinkingIndex = null
            }
            currentReasoningItemId = undefined
            currentReasoningItemEncrypted = undefined
          }
        }

        if (thinkingDelta && eventType.includes('reasoning')) {
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
            delta: { type: 'thinking_delta', thinking: thinkingDelta },
          } as BetaRawMessageStreamEvent
          emittedAnyContent = true
        } else if (
          textDelta &&
          (eventType.includes('output_text') ||
            (eventType.includes('content_part') && !eventType.includes('reasoning')))
        ) {
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
            delta: { type: 'text_delta', text: textDelta },
          } as BetaRawMessageStreamEvent
          emittedAnyContent = true
        }

        if (eventType.includes('function_call')) {
          const details = getToolCallDetails(event)
          const toolId = details.id ?? `toolu_responses_${toolIdsInOrder.length}`
          let toolState = toolStateById.get(toolId)
          if (!toolState) {
            toolState = { id: toolId, name: details.name ?? '', arguments: '' }
            toolStateById.set(toolId, toolState)
            toolIdsInOrder.push(toolId)
          }
          if (details.name) toolState.name = details.name

          const argumentsDelta =
            typeof event.arguments_delta === 'string'
              ? event.arguments_delta
              : eventType.includes('function_call_arguments') && typeof event.delta === 'string'
                ? event.delta
                : typeof event.item?.arguments === 'string' && eventType.includes('added')
                  ? (event.item.arguments as string)
                  : undefined
          if (argumentsDelta) {
            toolState.arguments += argumentsDelta
          }
          if (stopReason !== 'max_tokens') stopReason = 'tool_use'
        }

        if (eventType === 'response.completed' || eventType === 'response.incomplete') {
          if (
            event.response?.status === 'incomplete' &&
            event.response.incomplete_details?.reason === 'max_output_tokens'
          ) {
            stopReason = 'max_tokens'
          }
        }

        if (eventType === 'response.failed') {
          const err = event.response?.error
          const detail = err?.message || err?.type || event.response?.status || 'unknown'
          throw new Error(`[openaiResponsesCompat] response.failed: ${detail}`)
        }
      }
    }
  }

  if (!started) {
    throw new Error(
      `[openaiResponsesCompat] stream ended before message_start for model=${input.model}`,
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

  for (const toolId of toolIdsInOrder) {
    const toolState = toolStateById.get(toolId)
    if (!toolState) continue
    if (!toolState.name) {
      throw new Error(`[openaiResponsesCompat] function_call ${toolId} has no name`)
    }
    const idx = nextContentIndex++
    yield {
      type: 'content_block_start',
      index: idx,
      content_block: { type: 'tool_use', id: toolState.id, name: toolState.name, input: {} },
    } as BetaRawMessageStreamEvent
    yield {
      type: 'content_block_delta',
      index: idx,
      delta: { type: 'input_json_delta', partial_json: toolState.arguments || '{}' },
    } as BetaRawMessageStreamEvent
    yield { type: 'content_block_stop', index: idx } as BetaRawMessageStreamEvent
    emittedAnyContent = true
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

type OpenAIResponsesOutputItem = {
  type?: string
  id?: string
  role?: string
  call_id?: string
  name?: string
  arguments?: string
  content?: Array<{
    type?: string
    text?: string
  }>
  summary?: Array<{ text?: string }>
}

type OpenAIResponsesNonStreamingResponse = {
  id?: string
  status?: string
  output?: OpenAIResponsesOutputItem[]
  usage?: {
    input_tokens?: number
    output_tokens?: number
  }
  incomplete_details?: { reason?: string } | null
  error?: { message?: string; code?: string; type?: string }
}

export async function requestOpenAIResponsesNonStream(
  config: OpenAICompatConfig,
  request: OpenAIResponsesRequest,
  signal?: AbortSignal,
): Promise<OpenAIResponsesNonStreamingResponse> {
  const response = await (config.fetch ?? globalThis.fetch)(
    joinBaseUrl(config.baseURL, '/responses'),
    {
      method: 'POST',
      signal,
      headers: {
        'content-type': 'application/json',
        authorization: `Bearer ${config.apiKey}`,
        ...config.headers,
      },
      body: JSON.stringify({ ...request, stream: false }),
    },
  )

  const responseText = await response.text().catch(() => '')

  if (!response.ok) {
    throw new Error(
      `OpenAI Responses compatible request failed with status ${response.status}${responseText ? `: ${responseText}` : ''}`,
    )
  }

  let parsed: OpenAIResponsesNonStreamingResponse
  try {
    parsed = JSON.parse(responseText) as OpenAIResponsesNonStreamingResponse
  } catch {
    throw new Error(
      `OpenAI Responses endpoint returned non-JSON response: ${responseText.slice(0, 500)}`,
    )
  }

  if (parsed.error) {
    const msg =
      parsed.error.message ||
      parsed.error.type ||
      JSON.stringify(parsed.error).slice(0, 500)
    throw new Error(`[openaiResponsesCompat] non-streaming error: ${msg}`)
  }

  return parsed
}

export function createBetaMessageFromOpenAIResponsesResponse(input: {
  response: OpenAIResponsesNonStreamingResponse
  model: string
}): BetaMessage {
  type AnyBlock = Record<string, unknown>
  const content: AnyBlock[] = []
  let hasToolCall = false

  for (const item of input.response.output ?? []) {
    if (item.type === 'reasoning') {
      const thinkingText = (item.summary ?? [])
        .map(part => (typeof part?.text === 'string' ? part.text : ''))
        .join('')
      if (thinkingText.length > 0) {
        content.push({ type: 'thinking', thinking: thinkingText, signature: '' })
      }
      continue
    }

    if (item.type === 'message') {
      const text = (item.content ?? [])
        .filter(part => part?.type === 'output_text')
        .map(part => (typeof part?.text === 'string' ? part.text : ''))
        .join('')
      if (text.length > 0) {
        content.push({ type: 'text', text })
      }
      continue
    }

    if (item.type === 'function_call') {
      if (!item.name) {
        throw new Error('[openaiResponsesCompat] non-streaming function_call missing name')
      }
      const rawArgs = item.arguments ?? ''
      let parsedArgs: unknown = {}
      if (rawArgs.length > 0) {
        try {
          parsedArgs = JSON.parse(rawArgs)
        } catch {
          throw new Error(
            `[openaiResponsesCompat] non-streaming function_call arguments invalid JSON for tool=${item.name}`,
          )
        }
      }
      content.push({
        type: 'tool_use',
        id: item.call_id ?? item.id ?? `toolu_openai_responses_${content.length}`,
        name: item.name,
        input: parsedArgs,
      })
      hasToolCall = true
    }
  }

  if (content.length === 0) {
    content.push({ type: 'text', text: '' })
  }

  let stopReason: BetaMessage['stop_reason'] = 'end_turn'
  if (input.response.incomplete_details?.reason === 'max_output_tokens') {
    stopReason = 'max_tokens'
  } else if (hasToolCall) {
    stopReason = 'tool_use'
  }

  return {
    id: input.response.id ?? 'openai-responses-compat',
    type: 'message',
    role: 'assistant',
    model: input.model,
    content: content as unknown as BetaMessage['content'],
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: input.response.usage?.input_tokens ?? 0,
      output_tokens: input.response.usage?.output_tokens ?? 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    },
  } as BetaMessage
}
