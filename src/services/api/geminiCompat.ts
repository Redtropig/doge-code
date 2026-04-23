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

type AnyBlock = Record<string, unknown>

type GeminiPart = {
  text?: string
  thought?: boolean
  thoughtSignature?: string
  inlineData?: {
    mimeType: string
    data: string
  }
  functionCall?: {
    name?: string
    args?: unknown
  }
  functionResponse?: {
    name?: string
    response?: unknown
  }
}

type GeminiContent = {
  role: 'user' | 'model'
  parts: GeminiPart[]
}

type GeminiTool = {
  functionDeclarations: Array<{
    name: string
    description?: string
    parameters?: unknown
  }>
}

type GeminiRequest = {
  contents: GeminiContent[]
  systemInstruction?: {
    parts: Array<{ text: string }>
  }
  tools?: GeminiTool[]
  toolConfig?: {
    functionCallingConfig: {
      mode: 'AUTO' | 'ANY' | 'NONE'
      allowedFunctionNames?: string[]
    }
  }
  generationConfig?: {
    temperature?: number
    maxOutputTokens?: number
    thinkingConfig?: {
      includeThoughts?: boolean
      thinkingBudget?: number
    }
  }
}

type GeminiStreamChunk = {
  candidates?: Array<{
    content?: {
      parts?: GeminiPart[]
    }
    finishReason?: string
  }>
  usageMetadata?: {
    promptTokenCount?: number
    candidatesTokenCount?: number
    totalTokenCount?: number
  }
  promptFeedback?: {
    blockReason?: string
    blockReasonMessage?: string
  }
  error?: { code?: number; message?: string; status?: string }
}

function getToolNameById(messages: BetaMessageParam[]): Map<string, string> {
  const toolNameById = new Map<string, string>()

  for (const message of messages) {
    if (message.role !== 'assistant' || !Array.isArray(message.content)) continue
    for (const block of message.content as unknown as AnyBlock[]) {
      if (
        block.type === 'tool_use' &&
        typeof block.id === 'string' &&
        typeof block.name === 'string'
      ) {
        toolNameById.set(block.id, block.name)
      }
    }
  }

  return toolNameById
}

function getGeminiToolDefinitions(tools?: BetaToolUnion[]): GeminiTool[] | undefined {
  const definitions = getToolDefinitions(tools)
  if (!definitions || definitions.length === 0) return undefined

  return [
    {
      functionDeclarations: definitions.map(tool => ({
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters,
      })),
    },
  ]
}

function mapToolChoice(
  toolChoice?: BetaToolChoice,
): GeminiRequest['toolConfig'] | undefined {
  if (toolChoice?.type === 'tool') {
    return {
      functionCallingConfig: {
        mode: 'ANY',
        allowedFunctionNames: [toolChoice.name],
      },
    }
  }

  if (toolChoice?.type === 'auto') {
    return { functionCallingConfig: { mode: 'AUTO' } }
  }

  if (toolChoice?.type === 'any') {
    return { functionCallingConfig: { mode: 'ANY' } }
  }

  if (toolChoice?.type === 'none') {
    return { functionCallingConfig: { mode: 'NONE' } }
  }

  return undefined
}

function mapEffortToGeminiThinkingBudget(effort?: EffortValue): number | undefined {
  if (effort === 'none') return 0
  if (effort === 'low') return 1024
  if (effort === 'medium') return 4096
  if (effort === 'high') return 8192
  if (effort === 'max' || typeof effort === 'number') return 8192
  return undefined
}

function mapAnthropicUserBlocksToGeminiParts(blocks: AnyBlock[]): GeminiPart[] {
  return blocks.flatMap(block => {
    if (block.type === 'text' && typeof block.text === 'string' && block.text.length > 0) {
      return [{ text: block.text }]
    }
    if (block.type === 'image' && block.source && typeof block.source === 'object') {
      const source = block.source as Record<string, unknown>
      if (
        source.type === 'base64' &&
        typeof source.media_type === 'string' &&
        typeof source.data === 'string'
      ) {
        return [{
          inlineData: {
            mimeType: String(source.media_type),
            data: String(source.data),
          },
        }]
      }
      if (source.type === 'url' && typeof source.url === 'string') {
        return [{ text: `[image: ${String(source.url)}]` }]
      }
    }
    if (block.type === 'document' && block.source && typeof block.source === 'object') {
      const source = block.source as Record<string, unknown>
      if (source.type === 'text' && typeof source.data === 'string') {
        return [{ text: String(source.data) }]
      }
    }
    return []
  })
}

export function convertAnthropicRequestToGemini(input: {
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
}): GeminiRequest {
  const toolNameById = getToolNameById(input.messages)
  const contents: GeminiContent[] = []

  for (const message of input.messages) {
    const blocks = toBlocks(message.content)

    if (message.role === 'user') {
      const parts: GeminiPart[] = []

      for (const block of blocks as AnyBlock[]) {
        if (block.type === 'tool_result') {
          if (typeof block.tool_use_id !== 'string' || block.tool_use_id.length === 0) {
            throw new Error('[geminiCompat] tool_result missing tool_use_id — cannot resolve function name')
          }
          const toolUseId = block.tool_use_id
          const toolName = toolNameById.get(toolUseId)
          if (!toolName) {
            throw new Error(`[geminiCompat] tool_result references unknown tool_use_id=${toolUseId}`)
          }
          const rawContent = block.content
          let textContent: string
          if (typeof rawContent === 'string') {
            textContent = rawContent
          } else if (Array.isArray(rawContent)) {
            textContent = (rawContent as AnyBlock[])
              .map(b => {
                if (b.type === 'text' && typeof b.text === 'string') return b.text
                return JSON.stringify(b)
              })
              .join('\n')
          } else {
            textContent = JSON.stringify(rawContent ?? '')
          }
          if (block.is_error === true) textContent = `[tool_error] ${textContent}`
          parts.push({
            functionResponse: {
              name: toolName,
              response: { content: textContent },
            },
          })
        }
      }

      parts.push(
        ...mapAnthropicUserBlocksToGeminiParts(
          blocks.filter(block => block.type !== 'tool_result') as AnyBlock[],
        ),
      )

      if (parts.length > 0) {
        contents.push({ role: 'user', parts })
      }
      continue
    }

    const parts: GeminiPart[] = []
    const assistantBlocks = Array.isArray(message.content)
      ? (message.content as unknown as AnyBlock[])
      : []

    for (const block of assistantBlocks) {
      if (block.type === 'text' && typeof block.text === 'string' && block.text.length > 0) {
        parts.push({ text: block.text })
        continue
      }

      if (block.type === 'tool_use') {
        if (typeof block.name !== 'string' || block.name.length === 0) {
          throw new Error('[geminiCompat] tool_use missing name — cannot build functionCall')
        }
        let args: unknown = block.input ?? {}
        if (typeof args === 'string') {
          try {
            args = args.length > 0 ? JSON.parse(args) : {}
          } catch {
            throw new Error(`[geminiCompat] tool_use.input is a string but not valid JSON for tool=${block.name}`)
          }
        }
        parts.push({
          functionCall: {
            name: block.name,
            args,
          },
        })
      }
    }

    if (parts.length > 0) {
      contents.push({ role: 'model', parts })
    }
  }

  const systemText = input.system
    ? Array.isArray(input.system)
      ? input.system.map(block => block.text ?? '').join('\n')
      : input.system
    : ''

  const thinkingBudget = mapEffortToGeminiThinkingBudget(input.effort)
  const thinkingEnabled =
    (input.thinking?.type === 'enabled' || input.thinking?.type === 'adaptive') &&
    thinkingBudget !== 0

  const coalescedContents: GeminiContent[] = []
  for (const entry of contents) {
    const prev = coalescedContents[coalescedContents.length - 1]
    if (prev && prev.role === entry.role) {
      prev.parts.push(...entry.parts)
    } else {
      coalescedContents.push(entry)
    }
  }

  return {
    contents: coalescedContents,
    ...(systemText.trim()
      ? {
          systemInstruction: {
            parts: [{ text: systemText }],
          },
        }
      : {}),
    ...(getGeminiToolDefinitions(input.tools)
      ? { tools: getGeminiToolDefinitions(input.tools) }
      : {}),
    ...(mapToolChoice(input.tool_choice)
      ? { toolConfig: mapToolChoice(input.tool_choice) }
      : {}),
    generationConfig: {
      temperature: input.temperature,
      maxOutputTokens: input.max_tokens,
      ...(thinkingEnabled
        ? {
            thinkingConfig: {
              includeThoughts: true,
              ...(typeof thinkingBudget === 'number' && thinkingBudget > 0
                ? { thinkingBudget }
                : {}),
            },
          }
        : {}),
    },
  }
}

export async function createGeminiCompatStream(
  config: OpenAICompatConfig,
  model: string,
  request: GeminiRequest,
  signal?: AbortSignal,
): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  const response = await (config.fetch ?? globalThis.fetch)(
    joinBaseUrl(
      config.baseURL,
      `/models/${encodeURIComponent(model)}:streamGenerateContent?alt=sse`,
    ),
    {
      method: 'POST',
      signal,
      headers: {
        'content-type': 'application/json',
        'x-goog-api-key': config.apiKey,
        ...config.headers,
      },
      body: JSON.stringify(request),
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
      `Gemini request failed with status ${response.status}${responseText ? `: ${responseText}` : ''}`,
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
      `Gemini endpoint returned non-streaming response (content-type: ${contentType || 'unknown'}): ${responseText.slice(0, 500)}`,
    )
  }

  return response.body.getReader()
}

function mapGeminiFinishReason(reason: string | undefined): BetaMessage['stop_reason'] {
  if (reason === 'MAX_TOKENS') return 'max_tokens'
  return 'end_turn'
}

const GEMINI_ERROR_FINISH_REASONS = new Set([
  'SAFETY',
  'RECITATION',
  'BLOCKLIST',
  'PROHIBITED_CONTENT',
  'SPII',
  'MALFORMED_FUNCTION_CALL',
])

export async function* createAnthropicStreamFromGemini(input: {
  reader: ReadableStreamDefaultReader<Uint8Array>
  model: string
}): AsyncGenerator<BetaRawMessageStreamEvent, BetaMessage, void> {
  const decoder = new TextDecoder()
  let buffer = ''
  let started = false
  let currentTextIndex: number | null = null
  let currentThinkingIndex: number | null = null
  let thinkingSignature = ''
  let nextContentIndex = 0
  let promptTokens = 0
  let completionTokens = 0
  let emittedAnyContent = false
  let stopReason: BetaMessage['stop_reason'] = 'end_turn'
  let toolCounter = 0
  const bufferedToolCalls: Array<{ id: string; name: string; args: unknown }> = []

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
        const chunk = JSON.parse(data) as GeminiStreamChunk
        if (!chunk || typeof chunk !== 'object') {
          throw new Error(`[geminiCompat] invalid stream chunk: ${String(data).slice(0, 500)}`)
        }
        if (chunk.error) {
          const msg = chunk.error.message || chunk.error.status || JSON.stringify(chunk.error)
          throw new Error(`[geminiCompat] stream error: ${msg}`)
        }
        if (chunk.promptFeedback?.blockReason) {
          const reason = chunk.promptFeedback.blockReason
          const detail = chunk.promptFeedback.blockReasonMessage
          throw new Error(
            `[geminiCompat] prompt blocked: ${reason}${detail ? ` — ${detail}` : ''}`,
          )
        }

        if (chunk.usageMetadata) {
          promptTokens = chunk.usageMetadata.promptTokenCount ?? promptTokens
          completionTokens = chunk.usageMetadata.candidatesTokenCount ?? completionTokens
        }

        if (!started) {
          started = true
          yield {
            type: 'message_start',
            message: {
              id: 'gemini-compat',
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

        const candidate = chunk.candidates?.[0]
        const parts = candidate?.content?.parts ?? []

        for (const part of parts) {
          const isThought = part.thought === true
          const hasText = typeof part.text === 'string' && part.text.length > 0

          if (hasText && isThought) {
            if (currentTextIndex !== null) {
              yield { type: 'content_block_stop', index: currentTextIndex } as BetaRawMessageStreamEvent
              currentTextIndex = null
            }
            if (currentThinkingIndex === null) {
              currentThinkingIndex = nextContentIndex++
              if (typeof part.thoughtSignature === 'string') thinkingSignature = part.thoughtSignature
              yield {
                type: 'content_block_start',
                index: currentThinkingIndex,
                content_block: { type: 'thinking', thinking: '', signature: thinkingSignature },
              } as BetaRawMessageStreamEvent
            } else if (typeof part.thoughtSignature === 'string' && part.thoughtSignature !== thinkingSignature) {
              thinkingSignature = part.thoughtSignature
              yield {
                type: 'content_block_delta',
                index: currentThinkingIndex,
                delta: { type: 'signature_delta', signature: thinkingSignature },
              } as BetaRawMessageStreamEvent
            }
            yield {
              type: 'content_block_delta',
              index: currentThinkingIndex,
              delta: { type: 'thinking_delta', thinking: part.text as string },
            } as BetaRawMessageStreamEvent
            emittedAnyContent = true
            continue
          }

          if (hasText) {
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
              delta: { type: 'text_delta', text: part.text as string },
            } as BetaRawMessageStreamEvent
            emittedAnyContent = true
          }

          if (part.functionCall) {
            toolCounter += 1
            bufferedToolCalls.push({
              id: `toolu_gemini_${toolCounter}`,
              name: part.functionCall.name ?? '',
              args: part.functionCall.args ?? {},
            })
            if (stopReason !== 'max_tokens') stopReason = 'tool_use'
          }
        }

        if (candidate?.finishReason) {
          if (GEMINI_ERROR_FINISH_REASONS.has(candidate.finishReason)) {
            throw new Error(
              `[geminiCompat] generation stopped: ${candidate.finishReason}`,
            )
          }
          const mapped = mapGeminiFinishReason(candidate.finishReason)
          if (mapped === 'max_tokens' || stopReason !== 'tool_use') {
            stopReason = mapped
          }
        }
      }
    }
  }

  if (!started) {
    throw new Error(`[geminiCompat] stream ended before message_start for model=${input.model}`)
  }

  if (currentTextIndex !== null) {
    yield { type: 'content_block_stop', index: currentTextIndex } as BetaRawMessageStreamEvent
    currentTextIndex = null
  }
  if (currentThinkingIndex !== null) {
    yield { type: 'content_block_stop', index: currentThinkingIndex } as BetaRawMessageStreamEvent
    currentThinkingIndex = null
  }

  for (const tool of bufferedToolCalls) {
    if (!tool.name) {
      throw new Error('[geminiCompat] functionCall missing name in stream')
    }
    const idx = nextContentIndex++
    yield {
      type: 'content_block_start',
      index: idx,
      content_block: { type: 'tool_use', id: tool.id, name: tool.name, input: {} },
    } as BetaRawMessageStreamEvent
    yield {
      type: 'content_block_delta',
      index: idx,
      delta: { type: 'input_json_delta', partial_json: JSON.stringify(tool.args ?? {}) },
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
    usage: { output_tokens: completionTokens },
  } as BetaRawMessageStreamEvent

  yield { type: 'message_stop' } as BetaRawMessageStreamEvent

  return {
    id: 'gemini-compat',
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
